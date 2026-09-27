"""
app.py — BeniMeteo | Dashboard Principal Streamlit
Application météo intelligente pour la région Béni Mellal-Khénifra
- Données historiques Open-Meteo API
- Prédictions ML (SARIMA + XGBoost)
- Alertes zones vulnérables (données HCP douar)
- Intégration QGIS (GeoJSON + PyQGIS)

Lancement : streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
from datetime import datetime, date
import warnings
warnings.filterwarnings("ignore")

# Imports modules locaux
from src.data_loader        import (fetch_data, fetch_forecast, DEFAULT_LAT, DEFAULT_LON,
                                    get_ml_training_data, get_data_source_info)
from src.preprocessing      import preprocess, get_monthly_stats, get_annual_stats
from src.visualization      import (
    temperature_series, precipitation_bar, wind_rose,
    heatmap_monthly, forecast_chart, multi_variable_chart,
    vulnerability_map_plotly,
)
from src.alerts             import get_zones, compute_vulnerability_score, alert_level, zones_to_geojson
from src.alert_engine       import (
    compute_score_v2, alert_level_v2, generate_alerts_v2,
    get_forecast_conditions, generate_forecast_alerts,
    fetch_hopitaux_osm, compute_spi, get_drought_alert,
    get_seasonal_context, estimer_malades_chroniques,
    save_alerts, save_snapshot, load_alert_history, load_snapshots,
    init_alert_db, get_csv_paths, HOPITAUX_REGION,
)
from src.notifications      import load_config as load_notif_config, save_config as save_notif_config, test_email_connection, check_and_notify
from src.meteo_multipoints  import fetch_all_provinces, get_province_summary, compute_regional_scores_multipoint, PROVINCE_POINTS
from src.report_generator   import generate_alert_report, save_report_to_file, list_reports
from src.scheduler          import start_scheduler, get_scheduler_logs
from src.demographics  import (
    load_hcp_data, aggregate_by_province, aggregate_by_commune, aggregate_by_cercle,
    get_top_vulnerable_zones, compute_vulnerability_score_hcp, build_region_geojson,
    PROVINCE_INFO, PROVINCE_COLORS,
)
from src.models.evaluation import evaluate_model

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="BeniMeteo | Béni Mellal-Khénifra",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CSS — Design professionnel Atlas
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

* { font-family: 'Inter', sans-serif !important; }
code, pre { font-family: monospace !important; }

/* ── Fond principal : blanc/gris très clair ── */
.stApp {
    background: #F8F9FA !important;
    color: #1A1A2E !important;
}

/* ── Sidebar : blanc avec bordure légère ── */
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E0E4EA !important;
}
[data-testid="stSidebar"] * { color: #1A1A2E !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: #374151 !important; }

/* ── En-têtes ── */
h1, h2, h3 { color: #1A3D6B !important; font-weight: 600 !important; }

/* ── Cards metrics ── */
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #E0E4EA !important;
    border-radius: 10px !important;
    padding: 14px !important;
}
[data-testid="stMetricValue"] { color: #1A3D6B !important; font-size: 1.6rem !important; }
[data-testid="stMetricLabel"] { color: #6B7280 !important; }
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* ── Conteneur principal ── */
.main .block-container {
    padding: 1.5rem 2rem !important;
    background: #F8F9FA !important;
}

/* ── Widgets Streamlit ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: #FFFFFF !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 8px !important;
    color: #1A1A2E !important;
}
.stSlider [data-baseweb="slider"] { background: #E5E7EB !important; }
.stButton > button {
    background: #1A3D6B !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.2rem !important;
    transition: background 0.2s !important;
}
.stButton > button:hover { background: #2D5A8E !important; }

/* ── Tableaux ── */
[data-testid="stDataFrame"] {
    background: #FFFFFF !important;
    border: 1px solid #E0E4EA !important;
    border-radius: 10px !important;
}
.stDataFrame thead tr th {
    background: #1A3D6B !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
}
.stDataFrame tbody tr:nth-child(even) { background: #F3F4F6 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #FFFFFF !important;
    border-bottom: 2px solid #E0E4EA !important;
    border-radius: 8px 8px 0 0 !important;
}
.stTabs [data-baseweb="tab"] {
    color: #6B7280 !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    color: #1A3D6B !important;
    border-bottom: 3px solid #1A3D6B !important;
    background: transparent !important;
}

/* ── Alertes niveaux ── */
.alert-rouge {
    background: #FEE2E2 !important; color: #991B1B !important;
    border-left: 4px solid #DC2626 !important;
    padding: 12px 16px !important; border-radius: 0 8px 8px 0 !important;
    margin: 8px 0 !important; font-weight: 500 !important;
}
.alert-orange {
    background: #FEF3C7 !important; color: #92400E !important;
    border-left: 4px solid #F59E0B !important;
    padding: 12px 16px !important; border-radius: 0 8px 8px 0 !important;
    margin: 8px 0 !important; font-weight: 500 !important;
}
.alert-jaune {
    background: #FFF9C4 !important; color: #78630A !important;
    border-left: 4px solid #EAB308 !important;
    padding: 12px 16px !important; border-radius: 0 8px 8px 0 !important;
    margin: 8px 0 !important;
}
.alert-vert {
    background: #DCFCE7 !important; color: #166534 !important;
    border-left: 4px solid #22C55E !important;
    padding: 12px 16px !important; border-radius: 0 8px 8px 0 !important;
    margin: 8px 0 !important;
}

/* ── Info / warning / success ── */
[data-testid="stInfo"] {
    background: #EFF6FF !important;
    border: 1px solid #BFDBFE !important;
    color: #1E40AF !important;
    border-radius: 8px !important;
}
[data-testid="stWarning"] {
    background: #FFFBEB !important;
    border: 1px solid #FCD34D !important;
    color: #92400E !important;
    border-radius: 8px !important;
}
[data-testid="stSuccess"] {
    background: #F0FDF4 !important;
    border: 1px solid #86EFAC !important;
    color: #166534 !important;
    border-radius: 8px !important;
}
[data-testid="stError"] {
    background: #FFF1F2 !important;
    border: 1px solid #FECDD3 !important;
    color: #9F1239 !important;
    border-radius: 8px !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #E0E4EA !important;
    border-radius: 10px !important;
}

/* ── Radio / navigation ── */
[data-testid="stRadio"] label {
    color: #374151 !important;
    font-weight: 400 !important;
}
[data-testid="stRadio"] [aria-checked="true"] + span {
    color: #1A3D6B !important;
    font-weight: 600 !important;
}

/* ── Plotly charts fond blanc ── */
.js-plotly-plot .plotly { background: #FFFFFF !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #F1F3F5; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌦️ BeniMeteo")
    st.markdown("*Béni Mellal-Khénifra — Intelligence Climatique*")
    st.markdown("---")

    page = st.radio("Navigation", [
        "🏠 Tableau de Bord",
        "📊 Données Historiques",
        "🔮 Prédictions ML",
        "🚨 Alertes Prévisions",
        "🗺️ Zones & Alertes",
        "👥 Analyse Démographique",
        "📜 Historique Alertes",
        "🌍 Export QGIS",
        "⚙️ Saisie & Import",
        "📧 Notifications",
        "📄 Rapports PDF",
        "🔬 Anomalies Climatiques",
        "🔬 Calibration Scientifique",
        "💧 Besoins en Eau",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**📍 Zone géographique**")

    PROVINCES_COORDS = {
        "🌍 Toute la région (Béni Mellal)": (32.337, -6.349),
        "🏔 Azilal":                         (31.967, -6.572),
        "🌾 Béni Mellal":                    (32.337, -6.349),
        "🌿 Fqih Ben Saleh":                 (32.502, -6.687),
        "⛰ Khénifra":                        (32.934, -5.671),
        "🪨 Khouribga":                      (32.886, -6.906),
    }

    province_sel = st.selectbox(
        "Province",
        list(PROVINCES_COORDS.keys()),
        index=0,
        label_visibility="collapsed",
    )
    _default_lat, _default_lon = PROVINCES_COORDS[province_sel]

    with st.expander("⚙️ Coordonnées précises", expanded=False):
        st.caption("Ajustez manuellement si besoin")
        lat_input = st.number_input("Latitude",  -90.0,  90.0, _default_lat, step=0.001, format="%.4f")
        lon_input = st.number_input("Longitude", -180.0, 180.0, _default_lon, step=0.001, format="%.4f")
    if "lat_input" not in dir():
        lat_input = _default_lat
        lon_input = _default_lon

    st.markdown("---")
    st.markdown("**⚡ Simulation scénarios météo**")
    st.caption("Curseurs pour tester des événements extrêmes")

    # Curseur par défaut = 25°C (pas d'alerte) — l'utilisateur ajuste selon les prévisions réelles
    _temp_defaut = 25

    cond_temp     = st.slider("🌡 Température max (°C)", -5, 55, _temp_defaut,
                               help="Seuils réels région BM-Khénifra : "
                                    "< 35°C = VERT | 35-38°C = JAUNE | "
                                    "38-42°C = ORANGE | ≥ 42°C = ROUGE")
    cond_precip   = st.slider("🌧 Précipitations (mm)", 0, 100, 5,
                               help="< 20mm = VERT | 20-40mm = JAUNE | 40-60mm = ORANGE | ≥ 60mm = ROUGE")
    cond_wind     = st.slider("💨 Vent (km/h)", 0, 120, 15,
                               help="< 60km/h = VERT | 60-75 = JAUNE | 75-90 = ORANGE | ≥ 90 = ROUGE")
    cond_humidity = st.slider("💧 Humidité (%)", 10, 100, 35)
    cond_uv       = st.slider("☀️ UV Index", 0, 11, 7)
    conditions = dict(tmax=cond_temp, temp=cond_temp, tmin=cond_temp-12,
                      precip=cond_precip, wind=cond_wind,
                      humidity=cond_humidity, uv=cond_uv,
                      mois=__import__('datetime').date.today().month)

    # Indicateur visuel en temps réel
    if cond_temp >= 42:
        st.error(f"🔴 {cond_temp}°C ≥ 42°C → ROUGE — Canicule critique")
    elif cond_temp >= 38:
        st.warning(f"🟠 {cond_temp}°C entre 38-42°C → ORANGE — Forte chaleur")
    elif cond_temp >= 35:
        st.warning(f"🟡 {cond_temp}°C entre 35-38°C → JAUNE — Chaleur significative")
    else:
        st.caption(f"🟢 {cond_temp}°C < 35°C → VERT — Pas d'alerte chaleur")

    st.markdown("---")
    st.markdown("**📅 Période historique**")
    start_year = st.selectbox("Depuis", list(range(2010, 2025)), index=0)
    start_date = f"{start_year}-01-01"

    st.markdown("---")
    st.markdown("**📁 Données démographiques**")
    demo_upload = st.file_uploader("CSV HCP douar (.csv)", type=["csv", "xlsx"])
    if demo_upload:
        with open("/tmp/demo_upload.csv", "wb") as f:
            f.write(demo_upload.getbuffer())
        demo_path = "/tmp/demo_upload.csv"
    else:
        demo_path = None

    st.markdown("---")
    st.markdown("**📡 Cache données locales**")

    from src.data_loader import METEOSTAT_CACHE, OPENMETEO_CACHE
    import os as _os

    ms_ok = _os.path.exists(METEOSTAT_CACHE)
    om_ok = _os.path.exists(OPENMETEO_CACHE)

    if ms_ok:
        sz = _os.path.getsize(METEOSTAT_CACHE) // 1024
        st.caption(f"✅ Meteostat : {sz} KB — hors ligne OK")
    else:
        st.caption("⚠️ Meteostat : pas encore de cache")

    if om_ok:
        sz = _os.path.getsize(OPENMETEO_CACHE) // 1024
        st.caption(f"✅ Open-Meteo ERA5 : {sz} KB — hors ligne OK")
    else:
        st.caption("⚠️ Open-Meteo ERA5 : pas encore de cache")

    # Bibliothèque meteostat — gratuite, sans clé

    # Statut bibliothèque meteostat
    try:
        import meteostat as _ms_test
        st.caption("✅ meteostat lib installée (gratuit, sans clé)")
    except ImportError:
        st.caption("⚠️ pip install meteostat  (recommandé)")

    if st.button("⬇️ Télécharger et mettre en cache",
                  help="À faire UNE FOIS avec internet."):
        from src.data_loader import fetch_meteostat_for_ml, fetch_data, OPENMETEO_CACHE

        with st.spinner("⏳ Téléchargement Meteostat..."):
            df_ms  = fetch_meteostat_for_ml(lat_input, lon_input, years=12)
            src_ms = df_ms["source"].iloc[0] if "source" in df_ms.columns else "simulé"

        if any(k in str(src_ms) for k in ["meteostat","openmeteo","open-meteo"]):
            jul_ms = df_ms[df_ms.index.month==7]["tmax"].dropna()
            tmax_j = f"Tmax juillet = {jul_ms.mean():.1f}°C" if len(jul_ms)>0 else ""
            src_l  = "meteostat lib" if "lib" in src_ms else "cache/ERA5"
            st.success(f"✅ {src_l} : {len(df_ms):,} jours — {tmax_j}")
        else:
            st.warning(
                "⚠️ Meteostat indisponible\n\n"
                "→ Installez la bibliothèque : **pip install meteostat**\n"
                "→ Ou Open-Meteo ERA5 sera utilisé à la place"
            )

        st.info("⏳ Open-Meteo ERA5 (par tranches de 5 ans)...")
        with st.spinner("Téléchargement 2010→2026..."):
            df_om  = fetch_data(start_date="2010-01-01", lat=lat_input, lon=lon_input)
            src_om = df_om["source"].iloc[0] if "source" in df_om.columns else "simulé"

        if "open-meteo" in str(src_om):
            jul_om = df_om[df_om.index.month==7]["tmax"].dropna()
            tmax_j = f"Tmax juillet = {jul_om.mean():.1f}°C" if len(jul_om)>0 else ""
            st.success(f"✅ Open-Meteo ERA5 : {len(df_om):,} jours — {tmax_j}")
        elif "cache" in str(src_om):
            st.success(f"✅ Cache ERA5 : {len(df_om):,} jours")
        else:
            st.error("❌ Open-Meteo 429 — attendez 1 min et réessayez")

        if any(k in str(src_ms) for k in ["meteostat","open-meteo"]) or "cache" in str(src_om):
            st.info("✅ Cache prêt — ML hors ligne opérationnel.")
        st.rerun()

    if st.button("⬇️ Cache 5 provinces (ML complet)",
                  help="Télécharge l'historique de chaque province séparément"):
        from src.data_loader import fetch_meteostat_all_provinces
        with st.spinner("Téléchargement 5 provinces (meteostat lib)..."):
            prov_res = fetch_meteostat_all_provinces(years=12)
        ok = sum(1 for df in prov_res.values() if len(df) > 365)
        st.success(f"✅ {ok}/5 provinces téléchargées")
        for prov, df in prov_res.items():
            if "tmax" in df.columns:
                jul = df[df.index.month==7]["tmax"].dropna()
                tmax_j = f"{jul.mean():.1f}°C" if len(jul)>0 else "—"
                st.caption(f"  {prov}: {len(df):,}j | juillet={tmax_j}")
        st.rerun()

        st.markdown("---")
    st.caption(f"📍 {province_sel.split(' ', 1)[-1]}")
    st.caption(f"Lat {lat_input:.4f} | Lon {lon_input:.4f}")
    st.caption("Source : Open-Meteo · Meteostat · HCP Maroc")


# ═══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DONNÉES (cached)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner="📡 Chargement données Open-Meteo…")
def load_hist(start: str, lat: float, lon: float):
    df = fetch_data(start_date=start, lat=lat, lon=lon)
    return preprocess(df)

@st.cache_data(ttl=1800, show_spinner="🌤 Chargement prévisions…")
def load_forecast_api(lat: float, lon: float):
    return fetch_forecast(days=14, lat=lat, lon=lon)

@st.cache_data(ttl=86400, show_spinner="👥 Chargement données HCP…")
def load_hcp(path=None):
    return load_hcp_data(path)


df_hist   = load_hist(start_date, lat_input, lon_input)
df_fc_api = load_forecast_api(lat_input, lon_input)
df_hcp    = load_hcp(demo_path)
df_zones  = get_zones(df_hcp)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — TABLEAU DE BORD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Tableau de Bord":

    prov_label = province_sel.split(' ', 1)[-1]
    st.markdown(f"""
    <div class="header-banner">
        <h1 style="margin:0;font-size:2.2rem;">🌦️ BeniMeteo</h1>
        <p style="color:#7BB8D4;margin:6px 0 0;font-size:1rem;">
            Surveillance Météorologique — <b>{prov_label}</b>
            &nbsp;·&nbsp; Lat {lat_input:.3f} / Lon {lon_input:.3f}
        </p>
        <p style="color:rgba(232,213,183,0.6);margin:4px 0 0;font-size:0.85rem;">
            Open-Meteo · Meteostat · HCP RGPH 2024 · Intégration QGIS
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPIs 30 derniers jours ─────────────────────────────────────────────────
    last30 = df_hist.iloc[-30:]
    prev30 = df_hist.iloc[-60:-30]

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    with c1:
        t_now  = round(last30["temp_mean"].mean(), 1)
        t_prev = round(prev30["temp_mean"].mean(), 1)
        st.metric("🌡 T° Moy.", f"{t_now}°C", f"{t_now-t_prev:+.1f}°")
    with c2:
        st.metric("🔥 T° Max", f"{round(last30['tmax'].max(),1)}°C")
    with c3:
        st.metric("❄ T° Min", f"{round(last30['tmin'].min(),1)}°C")
    with c4:
        p_tot = round(last30["precip"].sum(), 1)
        st.metric("🌧 Précip.", f"{p_tot} mm")
    with c5:
        h_moy = round(last30["humidity_mean"].mean(), 1) if "humidity_mean" in last30.columns else "—"
        st.metric("💧 Humidité", f"{h_moy}%")
    with c6:
        if "wind_max" in last30.columns:
            st.metric("💨 Vent Max", f"{round(last30['wind_max'].max(),1)} km/h")
    with c7:
        scores = [compute_vulnerability_score(z.to_dict(), conditions)[0] for _, z in df_zones.iterrows()]
        max_sc = max(scores)
        lvl, emoji, _ = alert_level(max_sc)
        st.metric(f"{emoji} Alerte Max", lvl, f"Score {max_sc:.0f}/100")

    st.markdown("---")

    # ── Graphiques principaux ─────────────────────────────────────────────────
    col_l, col_r = st.columns([3, 1])
    with col_l:
        st.markdown("#### 📈 Températures — 12 derniers mois")
        st.plotly_chart(temperature_series(df_hist.iloc[-365:]), use_container_width=True)
    with col_r:
        st.markdown("#### 🌬 Rose des Vents")
        if "wind_dir" in df_hist.columns:
            st.plotly_chart(wind_rose(df_hist.iloc[-365:]), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 🌧 Précipitations mensuelles")
        st.plotly_chart(precipitation_bar(df_hist), use_container_width=True)
    with col_b:
        st.markdown("#### 📅 Prévisions 14 jours (API)")
        if df_fc_api is not None and len(df_fc_api) > 0:
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(x=df_fc_api.index, y=df_fc_api["tmax"],
                                         fill="tonexty", name="T Max",
                                         line=dict(color="#C4622D", width=1.5)))
            fig_fc.add_trace(go.Scatter(x=df_fc_api.index, y=df_fc_api["tmin"],
                                         fill="tozeroy", name="T Min",
                                         line=dict(color="#7BB8D4", width=1.5),
                                         fillcolor="rgba(123,184,212,0.1)"))
            if "humidity" in df_fc_api.columns:
                fig_fc.add_trace(go.Scatter(x=df_fc_api.index, y=df_fc_api["humidity"],
                                             name="Humidité %", yaxis="y2",
                                             line=dict(color="#4A7C59", width=1.2, dash="dot")))
            fig_fc.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#E8D5B7", height=290,
                margin=dict(l=40, r=10, t=20, b=40),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Humidité %"),
            )
            st.plotly_chart(fig_fc, use_container_width=True)
        else:
            st.info("Prévisions API non disponibles.")

    # ── Alertes zones rapides ─────────────────────────────────────────────────
    st.markdown("#### 🚨 Alertes Zones Vulnérables (conditions simulées)")
    zone_results = []
    for _, z in df_zones.iterrows():
        sc, vp, als = compute_vulnerability_score(z.to_dict(), conditions)
        lvl, em, css = alert_level(sc)
        zone_results.append((sc, z, vp, als, lvl, em, css))
    zone_results.sort(key=lambda x: x[0], reverse=True)

    cols_al = st.columns(min(4, len(zone_results)))
    for i, (sc, z, vp, als, lvl, em, css) in enumerate(zone_results[:4]):
        with cols_al[i % 4]:
            html = f'<div class="{css}"><b>{z["nom"]}</b><br>{em} <b>{lvl}</b><br>'
            html += f'<small>👥 {vp:,} vulnérables | Score: {sc}/100</small>'
            if als:
                html += f'<br><small style="opacity:0.8">↳ {als[0][0]}</small>'
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)

    # ── Tendance réchauffement ────────────────────────────────────────────────
    st.markdown("#### 📉 Tendance de Réchauffement — Béni Mellal")
    ann = get_annual_stats(df_hist)
    if "temp_mean_mean" in ann.columns:
        z_fit   = np.polyfit(ann["year"], ann["temp_mean_mean"], 1)
        trend_y = np.polyval(z_fit, ann["year"])
        total   = z_fit[0] * (ann["year"].max() - ann["year"].min())

        fig_tr = go.Figure()
        fig_tr.add_trace(go.Scatter(x=ann["year"], y=ann["temp_mean_mean"],
                                     mode="lines+markers", name="T° annuelle moy.",
                                     line=dict(color="#7BB8D4", width=2.5),
                                     marker=dict(size=6, color="#C4622D")))
        fig_tr.add_trace(go.Scatter(x=ann["year"], y=trend_y, mode="lines",
                                     name=f"Tendance ({z_fit[0]:+.3f}°/an)",
                                     line=dict(color="#C4622D", dash="dash", width=1.8)))

        # Canicules
        if "canicule_sum" in ann.columns:
            fig_tr.add_trace(go.Bar(x=ann["year"], y=ann["canicule_sum"],
                                     name="Jours canicule (≥40°C)",
                                     marker_color="rgba(214,40,40,0.4)",
                                     yaxis="y2"))
            fig_tr.update_layout(yaxis2=dict(overlaying="y", side="right",
                                              title="Jours canicule", showgrid=False))

        fig_tr.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#E8D5B7", height=290,
            margin=dict(l=50, r=20, t=20, b=40),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="°C"),
        )
        st.plotly_chart(fig_tr, use_container_width=True)
        st.markdown(
            f'<div class="info-box">📊 Réchauffement estimé : <b>+{total:.2f}°C</b> '
            f'| Tendance : <b>{z_fit[0]:+.3f}°C/an</b></div>',
            unsafe_allow_html=True)

    # ── Météo multi-provinces ─────────────────────────────────────────────────
    st.markdown("#### 🌍 Météo par Province — Aujourd'hui")
    @st.cache_data(ttl=1800, show_spinner="🌍 Chargement météo 5 provinces...")
    def load_multipoint():
        return fetch_all_provinces(days=3)

    prov_data_db = load_multipoint()
    prov_summary = get_province_summary(prov_data_db)

    if len(prov_summary) > 0:
        cols_mp = st.columns(len(prov_summary))
        color_map_prov = {"Azilal":"#2D8B57","Béni Mellal":"#2D5A8E",
                           "Fqih Ben Saleh":"#C4622D","Khénifra":"#7BB8D4","Khouribga":"#FCBF49"}
        for i, (_, row) in enumerate(prov_summary.iterrows()):
            with cols_mp[i]:
                prov_color = color_map_prov.get(row["province"], "#7BB8D4")
                st.markdown(
                    f'<div style="background:rgba(123,184,212,0.08);border:1px solid '
                    f'{prov_color}44;border-radius:8px;padding:10px;text-align:center;">'
                    f'<b style="color:{prov_color};font-size:12px">{row["province"]}</b><br>'
                    f'<span style="font-size:20px;font-weight:bold">{row["tmax"]}°</span>'
                    f'<span style="opacity:0.6;font-size:13px"> / {row["tmin"]}°</span><br>'
                    f'<small>🌧 {row["precip"]}mm · 💨 {row["wind"]}km/h</small><br>'
                    f'<small style="opacity:0.5">⛰ {row["altitude"]}m</small>'
                    f'</div>',
                    unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — DONNÉES HISTORIQUES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Données Historiques":
    st.markdown("## 📊 Données Historiques — Béni Mellal")
    st.caption(f"Source : Open-Meteo API | {start_date} → {date.today()} | {len(df_hist):,} jours")

    VAR_LABELS = {
        "temp_mean":    "🌡 Température Moyenne (°C)",
        "tmax":         "🔥 Température Max (°C)",
        "tmin":         "❄ Température Min (°C)",
        "temp_range":   "↕ Amplitude Thermique (°C)",
        "precip":       "🌧 Précipitations (mm)",
        "humidity_mean":"💧 Humidité Moy. (%)",
        "humidity_max": "💧 Humidité Max (%)",
        "wind_max":     "💨 Vent Max (km/h)",
        "wind_gusts":   "💨 Rafales (km/h)",
        "uv_max":       "☀️ UV Max",
        "pressure":     "📊 Pression (hPa)",
        "sunshine_h":   "🌞 Ensoleillement (h)",
        "heat_index":   "🥵 Indice Chaleur (°C)",
        "evapotransp":  "🌱 Évapotranspiration (mm)",
        "snowfall":     "❄ Neige (mm)",
    }
    avail = {k: v for k, v in VAR_LABELS.items() if k in df_hist.columns}

    c1, c2 = st.columns([1, 3])
    with c1:
        yr_range = st.slider("Années", int(df_hist["year"].min()), int(df_hist["year"].max()),
                             (int(df_hist["year"].min()), int(df_hist["year"].max())))
        var = st.selectbox("Variable", list(avail.keys()), format_func=lambda x: avail[x])
        agg = st.radio("Agrégation", ["Journalier", "Mensuel", "Annuel"], index=1)

    df_f = df_hist[(df_hist["year"] >= yr_range[0]) & (df_hist["year"] <= yr_range[1])]

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Série temporelle", "📆 Climatologie", "🗺 Heatmap", "📊 Multi-variables", "📋 Données brutes"
    ])

    with tab1:
        dp = (df_f.resample("ME")[var].mean() if agg == "Mensuel"
              else df_f.resample("YE")[var].mean() if agg == "Annuel"
              else df_f[var])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dp.index, y=dp.values, mode="lines",
                                  line=dict(color="#7BB8D4", width=1.8),
                                  fill="tozeroy", fillcolor="rgba(123,184,212,0.08)"))
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#E8D5B7", height=380,
                          margin=dict(l=50, r=20, t=20, b=50),
                          xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                          yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title=avail[var]))
        st.plotly_chart(fig, use_container_width=True)
        s = df_f[var].dropna()
        kc1, kc2, kc3, kc4, kc5 = st.columns(5)
        kc1.metric("Min", f"{s.min():.1f}")
        kc2.metric("Max", f"{s.max():.1f}")
        kc3.metric("Moy.", f"{s.mean():.1f}")
        kc4.metric("Méd.", f"{s.median():.1f}")
        kc5.metric("σ", f"{s.std():.2f}")

    with tab2:
        MONTHS_FR = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"]
        dm = df_f.groupby("month")[var].agg(["mean","min","max","std"]).reset_index()
        dm["mois"] = [MONTHS_FR[m-1] for m in dm["month"]]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=dm["mois"], y=dm["mean"], name="Moyenne",
                               marker_color="#7BB8D4",
                               error_y=dict(type="data", array=dm["std"], visible=True, color="#C4622D")))
        fig2.add_trace(go.Scatter(x=dm["mois"], y=dm["max"], mode="lines+markers",
                                   name="Max", line=dict(color="#C4622D", dash="dot"), marker=dict(size=5)))
        fig2.add_trace(go.Scatter(x=dm["mois"], y=dm["min"], mode="lines+markers",
                                   name="Min", line=dict(color="#7BB8D4", dash="dot"), marker=dict(size=5)))
        fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           font_color="#E8D5B7", height=350,
                           margin=dict(l=50, r=20, t=20, b=40),
                           legend=dict(bgcolor="rgba(0,0,0,0)"),
                           xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                           yaxis=dict(gridcolor="rgba(255,255,255,0.06)"))
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.plotly_chart(heatmap_monthly(df_f, var, avail[var]), use_container_width=True)

    with tab4:
        st.plotly_chart(multi_variable_chart(df_hist.iloc[-365:]), use_container_width=True)

    with tab5:
        display_cols = ["tmax","tmin","temp_mean","precip","humidity_mean","wind_max",
                        "wind_gusts","uv_max","pressure","evapotransp","sunshine_h","heat_index"]
        display_cols = [c for c in display_cols if c in df_f.columns]
        st.dataframe(df_f[display_cols].tail(200), use_container_width=True)
        st.download_button("⬇️ Télécharger CSV complet", df_f.to_csv(),
                           "beni_mellal_historique.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — PRÉDICTIONS ML  (Meteostat → SARIMA + XGBoost)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prédictions ML":
    st.markdown("## 🔮 Prédictions Climatiques — SARIMA + XGBoost")
    st.markdown("""
    <div class="alert-orange">
    <b>📡 Architecture double source :</b>
    <b>Meteostat</b> (stations réelles) → entraînement SARIMA + XGBoost &nbsp;|&nbsp;
    <b>Open-Meteo</b> → affichage historique + prévisions 14 jours
    </div>
    """, unsafe_allow_html=True)

    # ── Configuration ──────────────────────────────────────────────────────────
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        forecast_steps = st.selectbox(
            "Horizon de prédiction",
            [30, 90, 180, 365],
            format_func=lambda x: {30:"30 jours (1 mois)",
                                    90:"90 jours (saison)",
                                   180:"180 jours (6 mois)",
                                   365:"365 jours (1 an)"}[x])
    with col_cfg2:
        model_choice = st.selectbox("Modèle",
            ["SARIMA", "XGBoost", "XGBoost Optimisé (GridSearch)",
             "Prophet", "LSTM (Deep Learning)",
             "Comparaison tous modèles", "Comparaison SARIMA vs XGBoost"])
    with col_cfg3:
        vars_available = [v for v in ["tmax","temp_mean","tmin","precip","humidity_mean","wind_max"]
                          if v in df_hist.columns]
        target_var = st.selectbox("Variable cible",
            vars_available,
            index=0,
            format_func=lambda v: {"temp_mean":"🌡 T° Moyenne","tmax":"🔥 T° Max (réaliste été)",
                                    "tmin":"❄ T° Min","precip":"🌧 Précip.",
                                    "humidity_mean":"💧 Humidité","wind_max":"💨 Vent"}.get(v,v),
            help="T° Max recommandé — montre les vraies températures estivales (40-48°C)")

    st.markdown("---")

    # ── Chargement données Meteostat pour ML ───────────────────────────────────
    @st.cache_data(ttl=86400, show_spinner=False)
    def load_ml_data(lat, lon):
        return get_ml_training_data(lat=lat, lon=lon, years=12)

    with st.spinner("📡 Chargement historique Meteostat pour ML..."):
        df_ml = load_ml_data(lat_input, lon_input)

    # Renommage si nécessaire pour compatibilité modèles
    if "tmean" in df_ml.columns and "temp_mean" not in df_ml.columns:
        df_ml["temp_mean"] = df_ml["tmean"]
    if "humidity_max" in df_ml.columns and "humidity_mean" not in df_ml.columns:
        df_ml["humidity_mean"] = df_ml["humidity_max"]

    # Indicateur source
    src_col = df_ml["source"].iloc[0] if "source" in df_ml.columns and len(df_ml) > 0 else "simulé"
    total_j = len(df_ml)
    is_real = any(k in src_col for k in ["meteostat", "open-meteo"])
    is_cache = "cache" in src_col

    c_s1, c_s2, c_s3, c_s4 = st.columns(4)
    c_s1.metric("📅 Jours historique ML", f"{total_j:,}")
    if "meteostat" in src_col:
        c_s2.metric("📡 Source", "Meteostat ✅" if not is_cache else "Cache local ✅")
    elif "openmeteo" in src_col or "open-meteo" in src_col:
        c_s2.metric("📡 Source", "Open-Meteo ERA5 ✅" if not is_cache else "Cache ERA5 ✅")
    else:
        c_s2.metric("📡 Source", "Simulation DMN ⚠️")
    # Tmax juillet pour vérification
    if "tmax" in df_ml.columns:
        jul = df_ml[df_ml.index.month == 7]["tmax"].dropna()
        c_s3.metric("🌡 Tmax juillet (hist.)", f"{jul.mean():.1f}°C" if len(jul) > 0 else "—",
                    help="Doit être ~43°C pour Béni Mellal")
    c_s4.metric("🎯 Variable cible", target_var)

    # Message selon la source utilisée
    if "meteostat" in src_col:
        st.success(f"✅ Données Meteostat (stations réelles) — Tmax juillet = "
                   f"{df_ml[df_ml.index.month==7]['tmax'].mean():.1f}°C" 
                   if "tmax" in df_ml.columns else "✅ Données Meteostat chargées")
    elif "openmeteo" in src_col or "open-meteo" in src_col:
        st.info("ℹ️ Open-Meteo ERA5 utilisé (Meteostat indisponible) — "
                "données quasi-équivalentes, légèrement lissées.")
    elif "simulé" in src_col:
        st.warning(
            "⚠️ **Données simulées (climatologie DMN)** — Meteostat et Open-Meteo sont "
            "inaccessibles.\n\n"
            "**Pour obtenir les vraies données de stations météo :**\n"
            "1. Assurez-vous que `meteostat` est installé : `pip install meteostat`\n"
            "2. Cliquez sur **'⬇️ Télécharger et mettre en cache'** dans la barre latérale\n"
            "3. Une fois le cache créé, les prévisions utiliseront les vraies températures "
            "(Tmax juillet ~43°C à Béni Mellal)"
        )

    st.markdown("---")
    sar_mean, xgb_pred = None, None

    # ── SARIMA ─────────────────────────────────────────────────────────────────
    if "SARIMA" in model_choice:
        with st.spinner(f"🧮 Entraînement SARIMA — {forecast_steps} jours..."):
            from src.models.sarima import train_sarima
            _, sar_mean, sar_ci = train_sarima(
                df_ml, target=target_var, steps=forecast_steps)

        if sar_mean is not None:
            lower = sar_ci.iloc[:, 0] if sar_ci is not None else None
            upper = sar_ci.iloc[:, 1] if sar_ci is not None else None
            st.plotly_chart(
                forecast_chart(df_ml[target_var], sar_mean, lower, upper, "SARIMA"),
                use_container_width=True)
            test_actual = df_ml[target_var].iloc[-min(forecast_steps, len(df_ml)//5):]
            if len(test_actual) >= 10:
                m = evaluate_model(test_actual,
                                   sar_mean.iloc[:len(test_actual)], "SARIMA")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("RMSE", str(m["RMSE"]))
                c2.metric("MAE",  str(m["MAE"]))
                c3.metric("MAPE", f"{m['MAPE']}%")
                c4.metric("R²",   str(m["R2"]))

    # ── XGBoost ────────────────────────────────────────────────────────────────
    if "XGBoost" in model_choice and "Optimisé" not in model_choice and "tous" not in model_choice:
        with st.spinner(f"🤖 Entraînement XGBoost — {forecast_steps} jours..."):
            from src.models.xgboost_model import train_xgboost
            _, xgb_pred, xgb_metrics = train_xgboost(
                df_ml, target=target_var, forecast_steps=forecast_steps)

        if xgb_pred is not None:
            st.plotly_chart(
                forecast_chart(df_ml[target_var], xgb_pred, model_name="XGBoost"),
                use_container_width=True)
            if xgb_metrics:
                cx1, cx2, cx3, cx4 = st.columns(4)
                cx1.metric("RMSE", xgb_metrics.get("RMSE", "—"))
                cx2.metric("MAE",  xgb_metrics.get("MAE",  "—"))
                cx3.metric("R²",   xgb_metrics.get("R2",   "—"))
                cx4.metric("MAPE", xgb_metrics.get("MAPE", "—"))

    # ── XGBoost Optimisé ───────────────────────────────────────────────────────
    if "Optimisé" in model_choice:
        st.info("⏳ GridSearch peut prendre 2-5 minutes selon la taille des données.")
        with st.spinner(f"🔬 XGBoost GridSearch — {forecast_steps} jours..."):
            from src.models.ml_advanced import train_xgboost_optimized
            _, xgb_pred, xgb_metrics = train_xgboost_optimized(
                df_ml, target=target_var, forecast_steps=forecast_steps, optimize=True)
        if xgb_pred is not None:
            st.plotly_chart(
                forecast_chart(df_ml[target_var], xgb_pred, model_name="XGBoost Optimisé"),
                use_container_width=True)
            if xgb_metrics:
                cx1, cx2, cx3, cx4 = st.columns(4)
                cx1.metric("RMSE", xgb_metrics.get("RMSE", "—"))
                cx2.metric("MAE",  xgb_metrics.get("MAE",  "—"))
                cx3.metric("R²",   xgb_metrics.get("R2",   "—"))
                cx4.metric("Meilleurs params", str(xgb_metrics.get("best_params", {}))[:40])

    # ── Prophet ────────────────────────────────────────────────────────────────
    if "Prophet" in model_choice:
        with st.spinner(f"🔮 Entraînement Prophet — {forecast_steps} jours..."):
            from src.models.ml_advanced import train_prophet
            _, fc_tuple, prophet_metrics = train_prophet(
                df_ml, target=target_var, forecast_steps=forecast_steps)
        if fc_tuple is not None:
            fc_prop, conf_prop = fc_tuple
            lower_p = conf_prop["lower"] if conf_prop is not None else None
            upper_p = conf_prop["upper"] if conf_prop is not None else None
            st.plotly_chart(
                forecast_chart(df_ml[target_var], fc_prop, lower_p, upper_p, "Prophet"),
                use_container_width=True)
            if prophet_metrics:
                p1, p2 = st.columns(2)
                p1.metric("RMSE", prophet_metrics.get("RMSE", "—"))
                p2.metric("MAE",  prophet_metrics.get("MAE", "—"))
            xgb_pred = fc_prop  # Pour comparaison

    # ── LSTM ───────────────────────────────────────────────────────────────────
    if "LSTM" in model_choice:
        st.info("⏳ LSTM nécessite TensorFlow. Entraînement ~2-3 minutes.")
        with st.spinner(f"🧠 Entraînement LSTM — {forecast_steps} jours..."):
            from src.models.ml_advanced import train_lstm
            _, lstm_pred, lstm_metrics = train_lstm(
                df_ml, target=target_var, forecast_steps=forecast_steps)
        if lstm_pred is not None:
            st.plotly_chart(
                forecast_chart(df_ml[target_var], lstm_pred, model_name="LSTM"),
                use_container_width=True)
            if lstm_metrics:
                l1, l2 = st.columns(2)
                l1.metric("RMSE", lstm_metrics.get("RMSE", "—"))
                l2.metric("Modèle", "LSTM Deep Learning")
            xgb_pred = lstm_pred

    # ── Comparaison tous modèles ───────────────────────────────────────────────
    if "tous modèles" in model_choice:
        st.markdown("#### ⚖️ Comparaison tous modèles disponibles")
        with st.spinner("Entraînement de tous les modèles..."):
            from src.models.ml_advanced import compare_all_models
            all_results = compare_all_models(df_ml, target=target_var, steps=forecast_steps)

        fig_all = go.Figure()
        hist_tail = df_ml[target_var].iloc[-365:] if target_var in df_ml.columns else pd.Series()
        if len(hist_tail) > 0:
            fig_all.add_trace(go.Scatter(x=hist_tail.index, y=hist_tail,
                                          name="Historique", line=dict(color="#7BB8D4", width=1.5)))
        colors_ml = {"sarima":"#C4622D","xgboost":"#FCBF49","prophet":"#4A7C59","lstm":"#D62828"}
        for model_name, res in all_results.items():
            if "error" not in res and res.get("forecast") is not None:
                fc = res["forecast"]
                fig_all.add_trace(go.Scatter(x=fc.index, y=fc,
                    name=model_name.upper(),
                    line=dict(color=colors_ml.get(model_name,"#888"), dash="dash", width=2)))
        fig_all.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#E8D5B7", height=420,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            title="Comparaison tous modèles disponibles")
        st.plotly_chart(fig_all, use_container_width=True)

        # Tableau métriques
        metrics_rows = []
        for model_name, res in all_results.items():
            if "error" not in res:
                m = res.get("metrics", {})
                metrics_rows.append({
                    "Modèle": model_name.upper(),
                    "RMSE": m.get("RMSE","—"), "MAE": m.get("MAE","—"), "R²": m.get("R2","—")
                })
        if metrics_rows:
            st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True, hide_index=True)

    # ── Comparaison SARIMA vs XGBoost ──────────────────────────────────────────
    if "Comparaison SARIMA" in model_choice and sar_mean is not None and xgb_pred is not None:
        st.markdown("#### ⚖️ Comparaison SARIMA vs XGBoost")
        fig_cmp = go.Figure()
        hist_tail = df_ml[target_var].iloc[-365:]
        fig_cmp.add_trace(go.Scatter(x=hist_tail.index, y=hist_tail,
                                      name="Historique Meteostat",
                                      line=dict(color="#7BB8D4", width=2)))
        fig_cmp.add_trace(go.Scatter(x=sar_mean.index, y=sar_mean,
                                      name="SARIMA",
                                      line=dict(color="#C4622D", dash="dash", width=2)))
        fig_cmp.add_trace(go.Scatter(x=xgb_pred.index, y=xgb_pred,
                                      name="XGBoost",
                                      line=dict(color="#FCBF49", dash="dot", width=2)))
        fig_cmp.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#E8D5B7", height=400,
            margin=dict(l=50, r=20, t=20, b=40),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)"))
        st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Tableau prédictions mensuelles ─────────────────────────────────────────
    if sar_mean is not None:
        st.markdown("---")
        st.markdown("#### 📋 Prédictions Mensuelles")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("**SARIMA**")
            pm = sar_mean.resample("ME").mean().reset_index()
            pm.columns = ["Mois", f"{target_var} (SARIMA)"]
            pm[f"{target_var} (SARIMA)"] = pm[f"{target_var} (SARIMA)"].round(2)
            pm["Mois"] = pm["Mois"].dt.strftime("%b %Y")
            st.dataframe(pm, use_container_width=True, hide_index=True)
        if xgb_pred is not None:
            with col_t2:
                st.markdown("**XGBoost**")
                px2 = xgb_pred.resample("ME").mean().reset_index()
                px2.columns = ["Mois", f"{target_var} (XGBoost)"]
                px2[f"{target_var} (XGBoost)"] = px2[f"{target_var} (XGBoost)"].round(2)
                px2["Mois"] = px2["Mois"].dt.strftime("%b %Y")
                st.dataframe(px2, use_container_width=True, hide_index=True)

    st.info(
        "💡 **Meteostat** fournit l'historique des stations réelles pour un entraînement ML plus précis. "
        "**SARIMA** capture la saisonnalité | **XGBoost** capture les patterns non-linéaires. "
        "Les prévisions 14 jours du tableau de bord restent basées sur Open-Meteo (physique atmosphérique).")




# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — ALERTES PRÉVISIONS (Open-Meteo + Score V2)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Alertes Prévisions":
    st.markdown("## 🚨 Alertes Prévisions")
    st.markdown("""
    <div class="alert-orange">
    <b>📡 Sources combinées :</b>
    <b>Open-Meteo</b> (prévisions 7 jours temps réel) +
    <b>Meteostat/SARIMA</b> (prévisions long terme 30-365 jours) |
    Seuils absolus : < 35°C 🟢 | 35-38°C 🟡 | 38-42°C 🟠 | ≥ 42°C 🔴
    </div>
    """, unsafe_allow_html=True)

    # Contexte saisonnier
    seasonal = get_seasonal_context()
    if seasonal["labels"]:
        for lbl in seasonal["labels"]:
            st.info(lbl)

    # SPI sécheresse
    spi_val = compute_spi(df_hist) if len(df_hist) > 90 else 0.0
    drought = get_drought_alert(spi_val)
    if drought:
        css_d = {"ROUGE":"alert-rouge","ORANGE":"alert-orange","JAUNE":"alert-jaune"}.get(drought[2],"alert-vert")
        st.markdown(f'<div class="{css_d}"><b>{drought[0]}</b> — {drought[1]}</div>', unsafe_allow_html=True)

    # Hôpitaux
    @st.cache_data(ttl=86400, show_spinner="🏥 Chargement hôpitaux OSM...")
    def load_hopitaux(lat, lon):
        return fetch_hopitaux_osm(lat, lon)
    hopitaux = load_hopitaux(lat_input, lon_input)

    # ── Sources combinées Open-Meteo + Meteostat ──────────────────────────────
    from src.alert_engine import get_combined_forecast, get_alert_level_from_tmax

    with st.spinner("📡 Chargement prévisions combinées (Open-Meteo + Meteostat)..."):
        combined = get_combined_forecast(lat_input, lon_input,
                                          days_short=7, df_hist=df_hist)

    forecast_list = combined["court_terme"]
    source_court  = combined["source_court"]
    source_long   = combined["source_long"]
    correction    = combined["correction"]
    df_long       = combined["long_terme"]

    # Indicateur sources
    c_src1, c_src2, c_src3 = st.columns(3)
    c_src1.metric("📡 Court terme (7j)", source_court.upper(),
                   "✅ Temps réel" if source_court == "open-meteo" else "⚠️ Indisponible")
    c_src2.metric("📊 Long terme (365j)", source_long.replace("sarima-","SARIMA + ").upper(),
                   "✅ Stations réelles" if "meteostat" in source_long else
                   "⚠️ Simulation DMN" if "simulation" in source_long else "✅ ERA5")
    c_src3.metric("🔧 Correction biais", f"{correction:+.1f}°C",
                   "Ajustement Open-Meteo ↔ Meteostat")

    if "simulation" in source_long:
        st.warning(
            "⚠️ SARIMA utilise la simulation DMN — pour les vraies données stations : "
            "`pip install meteostat` puis cliquez **'⬇️ Télécharger et mettre en cache'**"
        )

    # Prévisions jour par jour (Open-Meteo)
    st.markdown("### 📅 Prévisions météo 7 jours (Open-Meteo temps réel)")
    if forecast_list:
        cols_fc = st.columns(len(forecast_list))
        for i, fc in enumerate(forecast_list):
            with cols_fc[i]:
                dt_label = pd.to_datetime(fc["date"]).strftime("%d/%m")
                from src.demographics import aggregate_by_commune
                comm_agg = aggregate_by_commune(df_hcp)
                scores_day = comm_agg.apply(
                    lambda r: compute_score_v2(r.to_dict(), fc, hopitaux, spi_val, seasonal), axis=1)
                max_score = scores_day.max()
                lvl, em, css = alert_level_v2(max_score)
                # Niveau direct depuis température
                lvl_direct, em_direct = get_alert_level_from_tmax(fc["tmax"])
                st.markdown(
                    f'<div class="{css}" style="text-align:center;padding:8px 4px;">'
                    f'<b>{dt_label}</b><br>{em}<br>'
                    f'<small>🌡{fc["tmax"]}°C | 🌧{fc["precip"]}mm</small><br>'
                    f'<small>T°: {em_direct} | Score: {max_score:.0f}</small></div>',
                    unsafe_allow_html=True)
    else:
        st.warning("⚠️ Prévisions Open-Meteo indisponibles — vérifiez votre connexion internet")

    # Prévisions long terme SARIMA (Meteostat)
    if df_long is not None and len(df_long) > 0:
        st.markdown("### 📈 Prévisions long terme SARIMA (Meteostat)")
        st.caption(f"Source: {source_long} | Correction biais: {correction:+.1f}°C")

        # Ajouter correction biais
        df_long_corr = df_long.copy()
        df_long_corr["tmax_prevu"] = df_long_corr["tmax_prevu"] + correction
        df_long_corr["ic_bas"]     = df_long_corr["ic_bas"]     + correction
        df_long_corr["ic_haut"]    = df_long_corr["ic_haut"]    + correction

        # Ajouter niveau d'alerte
        df_long_corr["niveau"] = df_long_corr["tmax_prevu"].apply(
            lambda t: get_alert_level_from_tmax(t)[1] + " " + get_alert_level_from_tmax(t)[0])

        # Résumé par mois
        df_long_corr["mois"] = df_long_corr.index.month
        noms_m = {1:"Janv",2:"Fév",3:"Mars",4:"Avr",5:"Mai",6:"Juin",
                   7:"Juil",8:"Août",9:"Sept",10:"Oct",11:"Nov",12:"Déc"}
        monthly = df_long_corr.groupby("mois").agg(
            tmax_moy=("tmax_prevu","mean"),
            tmax_max=("tmax_prevu","max"),
            ic_bas_moy=("ic_bas","mean"),
            ic_haut_moy=("ic_haut","mean"),
        ).reset_index()
        monthly["mois_nom"] = monthly["mois"].map(noms_m)
        monthly["niveau"] = monthly["tmax_moy"].apply(
            lambda t: get_alert_level_from_tmax(t)[1] + " " + get_alert_level_from_tmax(t)[0])

        st.dataframe(
            monthly[["mois_nom","tmax_moy","tmax_max","ic_bas_moy","ic_haut_moy","niveau"]]
            .rename(columns={"mois_nom":"Mois","tmax_moy":"Tmax moy (°C)",
                              "tmax_max":"Tmax max (°C)","ic_bas_moy":"IC bas",
                              "ic_haut_moy":"IC haut","niveau":"Niveau alerte"}),
            use_container_width=True, hide_index=True)

        # Graphique combiné
        import plotly.graph_objects as go
        fig_comb = go.Figure()
        # SARIMA long terme
        fig_comb.add_trace(go.Scatter(
            x=df_long_corr.index, y=df_long_corr["tmax_prevu"],
            name="SARIMA (Meteostat)", line=dict(color="#C4622D", width=2)))
        fig_comb.add_trace(go.Scatter(
            x=df_long_corr.index.tolist() + df_long_corr.index.tolist()[::-1],
            y=df_long_corr["ic_haut"].tolist() + df_long_corr["ic_bas"].tolist()[::-1],
            fill="toself", fillcolor="rgba(196,98,45,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="IC 95% SARIMA"))
        # Open-Meteo court terme
        if forecast_list:
            om_dates = [pd.to_datetime(f["date"]) for f in forecast_list]
            om_tmax  = [f["tmax"] for f in forecast_list]
            fig_comb.add_trace(go.Scatter(
                x=om_dates, y=om_tmax, mode="markers+lines",
                name="Open-Meteo (réel)", marker=dict(color="#1A3D6B", size=8),
                line=dict(color="#1A3D6B", width=2, dash="dot")))
        # Seuils
        for seuil, color, label in [
            (42,"#D62828","≥42°C ROUGE"),
            (38,"#E07820","≥38°C ORANGE"),
            (35,"#9A8010","≥35°C JAUNE")]:
            fig_comb.add_hline(y=seuil, line_dash="dash",
                                line_color=color, opacity=0.6,
                                annotation_text=label,
                                annotation_position="right")
        fig_comb.update_layout(
            title="Prévisions Tmax — Open-Meteo (7j) + SARIMA/Meteostat (365j)",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#E8D5B7", height=400,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)",
                       title="Température max (°C)"))
        st.plotly_chart(fig_comb, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🗺️ Communes en alerte — par jour")

    day_sel = st.selectbox("Jour à analyser",
        [fc["date"] for fc in forecast_list],
        format_func=lambda d: pd.to_datetime(d).strftime("%A %d/%m/%Y"))

    fc_sel = next(fc for fc in forecast_list if fc["date"] == day_sel)

    # Calcul scores toutes communes pour ce jour
    comm_agg = aggregate_by_commune(df_hcp)
    comm_agg["score"]  = comm_agg.apply(
        lambda r: compute_score_v2(r.to_dict(), fc_sel, hopitaux, spi_val, seasonal), axis=1)
    comm_agg["niveau"] = comm_agg["score"].apply(lambda s: alert_level_v2(s)[0])
    comm_agg["emoji"]  = comm_agg["score"].apply(lambda s: alert_level_v2(s)[1])
    comm_agg = comm_agg.sort_values("score", ascending=False).reset_index(drop=True)

    # KPIs
    n_r = (comm_agg["niveau"]=="ROUGE").sum()
    n_o = (comm_agg["niveau"]=="ORANGE").sum()
    n_j = (comm_agg["niveau"]=="JAUNE").sum()
    n_v = (comm_agg["niveau"]=="VERT").sum()
    vp_r = comm_agg[comm_agg["niveau"]=="ROUGE"]["pop_vulnerable"].sum()
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("🔴 ROUGE",  n_r, f"{vp_r:,} vuln.")
    k2.metric("🟠 ORANGE", n_o)
    k3.metric("🟡 JAUNE",  n_j)
    k4.metric("🟢 VERT",   n_v)
    k5.metric("🌡 T° max", f"{fc_sel['tmax']}°C")

    # Carte
    color_map = {"ROUGE":"#D62828","ORANGE":"#F77F00","JAUNE":"#FCBF49","VERT":"#4A7C59"}
    fig_fc = go.Figure()
    for niv, col in color_map.items():
        sub = comm_agg[comm_agg["niveau"]==niv]
        if len(sub) == 0: continue
        fig_fc.add_trace(go.Scattermapbox(
            lat=sub["lat"], lon=sub["lon"], mode="markers",
            marker=dict(size=(sub["score"]/3+8).clip(upper=35), color=col, opacity=0.85),
            name=f"{niv} ({len(sub)})",
            hovertemplate="<b>%{customdata[0]}</b><br>Score: %{customdata[1]}/100<br>Vuln: %{customdata[2]:,}<extra></extra>",
            customdata=list(zip(sub["commune"], sub["score"].round(1), sub["pop_vulnerable"].astype(int))),
        ))
    # Hôpitaux sur la carte
    if hopitaux:
        fig_fc.add_trace(go.Scattermapbox(
            lat=[h["lat"] for h in hopitaux], lon=[h["lon"] for h in hopitaux],
            mode="markers+text",
            marker=dict(size=14, color="#FFFFFF", symbol="hospital"),
            text=[h["nom"] for h in hopitaux],
            textposition="top right",
            name="🏥 Hôpitaux", hovertemplate="<b>%{text}</b><extra></extra>",
        ))
    fig_fc.update_layout(
        mapbox=dict(style="carto-darkmatter", center=dict(lat=32.3, lon=-6.2), zoom=7.5),
        height=500, margin=dict(l=0,r=0,t=0,b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(bgcolor="rgba(0,0,0,0.5)", font_color="white"),
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # Top communes à risque avec alertes détaillées
    st.markdown("### 📋 Détail des communes à risque")
    top_risk = comm_agg[comm_agg["niveau"].isin(["ROUGE","ORANGE"])].head(15)
    for _, row in top_risk.iterrows():
        entity = row.to_dict()
        chron = estimer_malades_chroniques(
            int(row["population"]), int(row.get("pop_60_plus",0)), int(row.get("pop_moins_15",0)))
        lat_r, lon_r = float(row["lat"]), float(row["lon"])
        dist_h = min([abs(lat_r - h["lat"]) + abs(lon_r - h["lon"]) for h in hopitaux], default=999) * 111
        lvl, em, css = alert_level_v2(row["score"])
        with st.expander(
            f"{em} **{row['commune']}** [{row['province_fr']}] — Score {row['score']}/100 "
            f"| {int(row['pop_vulnerable']):,} vuln. | {chron['total_chroniques']:,} chroniques"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                **👥 Population exposée**
                - Total : **{int(row['population']):,}**
                - Vulnérables (enfants+âgés) : **{int(row['pop_vulnerable']):,}**
                - Maladies chroniques estimées : **{chron['total_chroniques']:,}**
                - Altitude : **{int(row.get('altitude',0))} m**
                """)
            with col_b:
                als = generate_alerts_v2(entity, fc_sel, hopitaux, spi_val, seasonal)
                for a in als[:4]:
                    css_a = {"ROUGE":"alert-rouge","ORANGE":"alert-orange","JAUNE":"alert-jaune"}.get(a[2],"alert-vert")
                    st.markdown(f'<div class="{css_a}" style="margin:3px 0;font-size:12px"><b>{a[0]}</b><br>{a[1]}</div>',
                                unsafe_allow_html=True)

    # Sauvegarde auto dans SQLite
    if len(comm_agg) > 0:
        save_snapshot(comm_agg.rename(columns={"niveau":"niveau_alerte"}), fc_sel)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — HISTORIQUE ALERTES (SQLite + CSV)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📜 Historique Alertes":
    st.markdown("## 📜 Historique des Alertes")
    init_alert_db()
    csv_paths = get_csv_paths()
    import os as _os

    # Indicateurs de stockage
    col_db1, col_db2, col_db3 = st.columns(3)
    with col_db1:
        sqlite_ok = _os.path.exists(csv_paths["sqlite"])
        st.markdown(
            f'<div class="{"alert-vert" if sqlite_ok else "alert-orange"}">'
            f'🗄️ <b>SQLite</b> — {"✅ Actif" if sqlite_ok else "⚠️ Vide"}</div>',
            unsafe_allow_html=True)
    with col_db2:
        csv_ok = _os.path.exists(csv_paths["alertes"])
        st.markdown(
            f'<div class="{"alert-vert" if csv_ok else "alert-orange"}">'
            f'📄 <b>CSV Alertes</b> — {"✅ Actif" if csv_ok else "⚠️ Vide"}</div>',
            unsafe_allow_html=True)
    with col_db3:
        csv_snap_ok = _os.path.exists(csv_paths["snapshots"])
        st.markdown(
            f'<div class="{"alert-vert" if csv_snap_ok else "alert-orange"}">'
            f'📄 <b>CSV Snapshots</b> — {"✅ Actif" if csv_snap_ok else "⚠️ Vide"}</div>',
            unsafe_allow_html=True)

    st.caption("📁 Fichiers dans : `BeniMeteo/data/`  —  `alerts_history.db` · `alerts_history.csv` · `snapshots_history.csv`")
    st.markdown("---")

    tab_h1, tab_h2, tab_h3 = st.tabs(["📈 Évolution région", "🔍 Recherche commune", "📥 Télécharger"])

    with tab_h1:
        days_h = st.slider("Période (jours)", 7, 365, 90)
        snaps = load_snapshots(days_h)
        if len(snaps) > 0:
            snaps["timestamp"] = pd.to_datetime(snaps["timestamp"])
            fig_hist = go.Figure()
            for col_n, color, label in [
                ("nb_rouge",  "#D62828", "🔴 ROUGE"),
                ("nb_orange", "#F77F00", "🟠 ORANGE"),
                ("nb_jaune",  "#FCBF49", "🟡 JAUNE"),
                ("nb_vert",   "#4A7C59", "🟢 VERT"),
            ]:
                if col_n in snaps.columns:
                    fig_hist.add_trace(go.Scatter(
                        x=snaps["timestamp"], y=snaps[col_n],
                        name=label, line=dict(color=color, width=2)))
            fig_hist.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#E8D5B7", height=350,
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Nb communes"),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                title="Évolution des niveaux d'alerte dans le temps",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # Stats rapides
            ks1, ks2, ks3 = st.columns(3)
            ks1.metric("Snapshots enregistrés", len(snaps))
            ks2.metric("Max communes ROUGE", int(snaps["nb_rouge"].max()) if "nb_rouge" in snaps else 0)
            ks3.metric("Max pop. vulnérable ROUGE", f"{int(snaps['pop_rouge'].max()):,}" if "pop_rouge" in snaps else "0")
        else:
            st.info("Aucun historique disponible. Visitez **🚨 Alertes Prévisions** pour générer des données.")

    with tab_h2:
        comm_search = st.text_input("Rechercher une commune (arabe ou français)")
        days_c = st.slider("Période", 7, 180, 30, key="hist_days")
        df_hist_al = load_alert_history(days_c, comm_search if comm_search else None)
        if len(df_hist_al) > 0:
            # Résumé
            st.markdown(f"**{len(df_hist_al)} alertes trouvées**")
            niveau_counts = df_hist_al["niveau"].value_counts()
            kc1, kc2, kc3, kc4 = st.columns(4)
            kc1.metric("🔴 ROUGE",  niveau_counts.get("ROUGE",  0))
            kc2.metric("🟠 ORANGE", niveau_counts.get("ORANGE", 0))
            kc3.metric("🟡 JAUNE",  niveau_counts.get("JAUNE",  0))
            kc4.metric("🟢 VERT",   niveau_counts.get("VERT",   0))

            cols_show = [c for c in ["timestamp","commune","province","score","niveau","temp","precip","pop_vulnerable"]
                         if c in df_hist_al.columns]
            st.dataframe(df_hist_al[cols_show], use_container_width=True, hide_index=True)

            # Stats par commune
            if "commune" in df_hist_al.columns:
                by_comm = df_hist_al.groupby("commune").agg(
                    nb_alertes=("score","count"),
                    score_max=("score","max"),
                    score_moy=("score","mean"),
                ).reset_index()
                by_comm["score_moy"] = by_comm["score_moy"].round(1)
                st.markdown("**Résumé par commune :**")
                st.dataframe(by_comm.sort_values("score_max", ascending=False),
                            use_container_width=True, hide_index=True)
        else:
            st.info("Aucune alerte trouvée. Visitez **🚨 Alertes Prévisions** pour générer des données.")

    with tab_h3:
        st.markdown("#### 📥 Télécharger les fichiers d'historique")

        col_dl1, col_dl2, col_dl3 = st.columns(3)

        with col_dl1:
            st.markdown("**🗄️ Export SQLite → CSV**")
            df_all = load_alert_history(365)
            if len(df_all) > 0:
                csv_exp = df_all.to_csv(index=False, sep=";", encoding="utf-8-sig")
                st.download_button(
                    "⬇️ Alertes (SQLite → CSV)",
                    csv_exp.encode("utf-8-sig"),
                    f"alertes_benimeteo_{date.today()}.csv",
                    "text/csv")
                st.caption(f"{len(df_all):,} alertes")
            else:
                st.info("Aucune alerte SQLite")

        with col_dl2:
            st.markdown("**📄 Fichier CSV alertes**")
            if _os.path.exists(csv_paths["alertes"]):
                with open(csv_paths["alertes"], "rb") as f:
                    st.download_button(
                        "⬇️ alerts_history.csv",
                        f.read(),
                        "alerts_history.csv",
                        "text/csv")
                sz = _os.path.getsize(csv_paths["alertes"])
                st.caption(f"{sz // 1024} KB")
            else:
                st.info("Fichier CSV vide")

        with col_dl3:
            st.markdown("**📄 Fichier CSV snapshots**")
            if _os.path.exists(csv_paths["snapshots"]):
                with open(csv_paths["snapshots"], "rb") as f:
                    st.download_button(
                        "⬇️ snapshots_history.csv",
                        f.read(),
                        "snapshots_history.csv",
                        "text/csv")
                sz = _os.path.getsize(csv_paths["snapshots"])
                st.caption(f"{sz // 1024} KB")
            else:
                st.info("Fichier CSV vide")

        st.markdown("---")
        st.markdown("**📁 Emplacement local des fichiers :**")
        st.code(f"""
BeniMeteo/data/
├── alerts_history.db       ← Base SQLite principale
├── alerts_history.csv      ← Export CSV alertes (append automatique)
└── snapshots_history.csv   ← Export CSV snapshots région
        """)



# ═══════════════════════════════════════════════════════════════════════════════

# PAGE — ZONES & ALERTES — Cercle → Commune → Douar
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Zones & Alertes":
    st.markdown("## 🗺️ Zones & Alertes — Région Béni Mellal-Khénifra")

    from src.alert_engine import compute_score_v2 as compute_score, generate_alerts_v2 as generate_alerts, alert_level_v2 as _al
    from src.demographics import aggregate_by_cercle

    if df_hcp is None or len(df_hcp) == 0:
        st.warning("Données HCP non disponibles.")
        st.stop()

    # ── Charger les vraies prévisions ou utiliser les curseurs ────────────────
    col_src1, col_src2 = st.columns([3,1])
    with col_src1:
        use_real = st.checkbox(
            "📡 Utiliser les vraies prévisions Open-Meteo (recommandé)",
            value=True,
            help="Si coché : les scores sont calculés avec les vraies températures prévues. "
                 "Si décoché : utilisez les curseurs de la barre latérale pour simuler des scénarios.")

    if use_real:
        with st.spinner("📡 Chargement prévisions Open-Meteo..."):
            from src.alert_engine import get_forecast_conditions
            fc_list_zones = get_forecast_conditions(lat_input, lon_input, days=1)

        if fc_list_zones:
            fc_zones = fc_list_zones[0]
            tmax_reel = fc_zones.get("tmax", cond_temp)
            cond_zones = fc_zones
            cond_zones["mois"] = __import__('datetime').date.today().month
            from src.alert_engine import get_alert_level_from_tmax
            lvl_direct, em_direct = get_alert_level_from_tmax(tmax_reel)
            st.success(f"✅ Prévisions réelles chargées — Aujourd'hui : Tmax **{tmax_reel}°C** "
                       f"| Précip. {fc_zones.get('precip',0)}mm "
                       f"| Vent {fc_zones.get('wind',10)}km/h "
                       f"→ {em_direct} **{lvl_direct}**")
        else:
            st.warning("⚠️ Open-Meteo inaccessible — curseurs utilisés")
            fc_zones = conditions
            cond_zones = conditions
    else:
        fc_zones = conditions
        cond_zones = conditions
        tmax_reel = cond_temp
        st.info(f"⚙️ Mode simulation — T° {cond_temp}°C (curseur barre latérale)")

    # ── Bannière conditions actives ───────────────────────────────────────────
    t_affich = cond_zones.get("tmax", cond_zones.get("temp", cond_temp))
    st.markdown(f"""
    <div class="alert-orange">
    <b>📊 Conditions utilisées pour les scores :</b>
    T° <b>{t_affich}°C</b> | Précip. <b>{cond_zones.get('precip',0)} mm</b> |
    Vent <b>{cond_zones.get('wind',10)} km/h</b> |
    Humidité <b>{cond_zones.get('humidity',50)}%</b> | UV <b>{cond_zones.get('uv',5)}</b>
    </div>
    """, unsafe_allow_html=True)

    # ── Calcul scores cercle / commune / douar ─────────────────────────────────
    cerc_df = aggregate_by_cercle(df_hcp)
    cerc_df["score"] = cerc_df.apply(
        lambda r: compute_score(r.to_dict(), cond_zones), axis=1)
    cerc_df["niveau"] = cerc_df["score"].apply(lambda s: _al(s)[0])
    cerc_df["emoji"]  = cerc_df["score"].apply(lambda s: _al(s)[1])
    cerc_df = cerc_df.sort_values("score", ascending=False).reset_index(drop=True)

    # ── KPIs ───────────────────────────────────────────────────────────────────
    n_rouge  = (cerc_df["score"] >= 48).sum()
    n_orange = ((cerc_df["score"] >= 23) & (cerc_df["score"] < 48)).sum()
    n_jaune  = ((cerc_df["score"] >= 12) & (cerc_df["score"] < 23)).sum()
    n_vert   = (cerc_df["score"] < 12).sum()

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("📍 Zones (Cercles)",     f"{len(cerc_df)}")
    k2.metric("🏡 Communes",            f"{df_hcp['commune'].nunique()}")
    k3.metric("🔵 Douars",              f"{len(df_hcp):,}")
    k4.metric("🔴 Zones ROUGE",         n_rouge)
    k5.metric("🟠 Zones ORANGE",        n_orange)
    k6.metric("⚠️ Personnes vulnérables", f"{df_hcp['pop_vulnerable'].sum():,}")

    st.markdown("---")

    tab_z1, tab_z2, tab_z3 = st.tabs([
        "🗺 Carte des 19 zones", "📋 Zones → Communes → Douars", "🏘 Top Douars critiques"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — Carte interactive (niveaux : cercle / commune / douar)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_z1:
        map_lvl = st.radio("Niveau d'affichage :",
            ["🔵 Zones / Cercles (19)", "🏡 Communes (119)", "🔵 Douars (2485)"],
            horizontal=True)

        if map_lvl.startswith("🔵 Zones"):
            map_df   = cerc_df.copy()
            name_col = "cercle"
            extra_cols = ["nb_communes", "nb_douars"]
        elif map_lvl.startswith("🏡"):
            from src.demographics import aggregate_by_commune
            map_df = aggregate_by_commune(df_hcp)
            map_df["score"]  = map_df.apply(lambda r: compute_score(r.to_dict(), conditions), axis=1)
            map_df["niveau"] = map_df["score"].apply(lambda s: _al(s)[0])
            map_df["emoji"]  = map_df["score"].apply(lambda s: _al(s)[1])
            name_col = "commune"
            extra_cols = ["cercle", "nb_douars"]
        else:
            map_df = df_hcp.copy()
            map_df["score"]  = map_df.apply(lambda r: compute_score(r.to_dict(), conditions), axis=1)
            map_df["niveau"] = map_df["score"].apply(lambda s: _al(s)[0])
            map_df["emoji"]  = map_df["score"].apply(lambda s: _al(s)[1])
            name_col = "douar"
            extra_cols = ["commune", "cercle"]

        color_map_alert = {"ROUGE": "#D62828", "ORANGE": "#F77F00",
                           "JAUNE": "#FCBF49", "VERT": "#4A7C59"}
        fig_map = go.Figure()
        for niv, col in color_map_alert.items():
            sub = map_df[map_df["niveau"] == niv]
            if len(sub) == 0:
                continue
            label_emoji = {"ROUGE":"🔴","ORANGE":"🟠","JAUNE":"🟡","VERT":"🟢"}[niv]
            sizes = (sub["score"] / 3 + 8).clip(upper=35)

            # build hover customdata
            alt_col = sub["altitude"].fillna(0).astype(int) if "altitude" in sub.columns else pd.Series([0]*len(sub), index=sub.index)
            cd = list(zip(
                sub[name_col].fillna("?"),
                sub["province_fr"].fillna("?"),
                sub["population"].astype(int),
                sub["pop_vulnerable"].astype(int),
                alt_col,
                sub["score"].round(1),
            ))

            fig_map.add_trace(go.Scattermapbox(
                lat=sub["lat"], lon=sub["lon"],
                mode="markers",
                marker=dict(size=sizes, color=col, opacity=0.85),
                name=f"{label_emoji} {niv} ({len(sub)})",
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Province : %{customdata[1]}<br>"
                    "Population : %{customdata[2]:,}<br>"
                    "Vulnérables : %{customdata[3]:,}<br>"
                    "Altitude : %{customdata[4]} m<br>"
                    "Score : <b>%{customdata[5]}/100</b><br>"
                    "<extra></extra>"
                ),
                customdata=cd,
            ))

        fig_map.update_layout(
            mapbox=dict(style="carto-darkmatter",
                        center=dict(lat=32.3, lon=-6.2), zoom=7.5),
            height=530, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(bgcolor="rgba(0,0,0,0.5)", font_color="white"),
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # Résumé alertes
        st.markdown("**Résumé des niveaux d'alerte :**")
        col_s = st.columns(4)
        niv_emoji = {"ROUGE":"🔴","ORANGE":"🟠","JAUNE":"🟡","VERT":"🟢"}
        for i, (niv, col) in enumerate(color_map_alert.items()):
            cnt = (map_df["niveau"] == niv).sum()
            vpop = map_df[map_df["niveau"]==niv]["pop_vulnerable"].sum()
            col_s[i].metric(f"{niv_emoji[niv]} {niv}", cnt, f"{vpop:,} vuln.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — Hiérarchie Zone → Commune → Douar
    # ══════════════════════════════════════════════════════════════════════════
    with tab_z2:
        # Filtre province
        prov_z = st.selectbox("📍 Province",
            ["Toutes"] + sorted(df_hcp["province_fr"].unique().tolist()), key="z2_prov")

        cerc_filtered = cerc_df if prov_z == "Toutes" else \
            cerc_df[cerc_df["province_fr"] == prov_z]

        st.markdown(f"**{len(cerc_filtered)} zones (cercles)** — triées par score d'alerte décroissant")

        for _, zone_row in cerc_filtered.iterrows():
            prov_name  = zone_row["province_fr"]
            cerc_name  = zone_row["cercle"]
            score_z    = zone_row["score"]
            niveau_z   = zone_row["niveau"]
            emoji_z    = zone_row["emoji"]
            pop_z      = int(zone_row["population"])
            vp_z       = int(zone_row["pop_vulnerable"])
            nb_comm    = int(zone_row["nb_communes"])
            nb_douars  = int(zone_row["nb_douars"])
            alt_z      = int(zone_row.get("altitude", 0))
            lvl_z, em_z, css_z = _al(score_z)

            # Alerte courte pour titre
            als_z = generate_alerts(zone_row.to_dict(), conditions)
            alert_preview = f" — ⚠️ {als_z[0][0]}" if als_z else ""

            with st.expander(
                f"{emoji_z} **Zone {cerc_name}** [{prov_name}]"
                f" · {nb_comm} communes · {nb_douars} douars"
                f" · {pop_z:,} hab · Score {score_z}/100{alert_preview}"
            ):
                # ── En-tête zone ──
                st.markdown(f"""
                <div class="{css_z}" style="margin-bottom:10px">
                <b>{em_z} {cerc_name}</b> — Province {prov_name} — {niveau_z}<br>
                👥 Population : <b>{pop_z:,}</b> &nbsp;|&nbsp;
                ⚠️ Vulnérables : <b>{vp_z:,}</b> &nbsp;|&nbsp;
                ⛰️ Altitude moy. : {alt_z} m &nbsp;|&nbsp;
                Score : <b>{score_z}/100</b>
                </div>
                """, unsafe_allow_html=True)

                # Alertes zone
                if als_z:
                    for a in als_z:
                        css_a = {"ROUGE":"alert-rouge","ORANGE":"alert-orange",
                                 "JAUNE":"alert-jaune"}.get(a[2],"alert-vert")
                        st.markdown(
                            f'<div class="{css_a}" style="margin:3px 0">'
                            f'<b>{a[0]}</b> — {a[1]}</div>',
                            unsafe_allow_html=True)

                st.markdown("---")

                # ── Communes de cette zone ──
                df_zone = df_hcp[
                    (df_hcp["province_fr"] == prov_name) &
                    (df_hcp["cercle"] == cerc_name)
                ]
                communes_zone = sorted(df_zone["commune"].unique())
                st.markdown(f"#### 🏡 {len(communes_zone)} Communes")

                for comm_name in communes_zone:
                    df_comm = df_zone[df_zone["commune"] == comm_name]
                    pop_c   = int(df_comm["population"].sum())
                    vp_c    = int(df_comm["pop_vulnerable"].sum())
                    alt_c   = int(df_comm["altitude"].mean())
                    nb_d    = len(df_comm)
                    pct_m15 = df_comm["pop_moins_15"].sum() / pop_c * 100 if pop_c else 0
                    pct_60  = df_comm["pop_60_plus"].sum()  / pop_c * 100 if pop_c else 0

                    comm_ent = {
                        "population": pop_c, "pop_vulnerable": vp_c,
                        "pct_moins_15": pct_m15, "pct_60_plus": pct_60,
                        "altitude": alt_c,
                    }
                    sc_c  = compute_score(comm_ent, conditions)
                    als_c = generate_alerts(comm_ent, conditions)
                    lvl_c, em_c, css_c = _al(sc_c)

                    with st.expander(
                        f"{em_c} **{comm_name}** · {nb_d} douars"
                        f" · {pop_c:,} hab · {vp_c:,} vulnérables"
                        f" · Alt {alt_c}m · Score {sc_c}/100"
                    ):
                        col_info, col_alerts = st.columns([3, 2])

                        with col_info:
                            st.markdown(f"""
                            **📊 Démographie (HCP)**
                            | Indicateur | Valeur |
                            |---|---|
                            | Population totale | **{pop_c:,}** |
                            | Enfants < 15 ans | **{int(df_comm['pop_moins_15'].sum()):,}** ({pct_m15:.1f}%) |
                            | Adultes 15–59 ans | **{int(df_comm['pop_15_59'].sum()):,}** |
                            | Personnes âgées 60+ | **{int(df_comm['pop_60_plus'].sum()):,}** ({pct_60:.1f}%) |
                            | **Total vulnérables** | **{vp_c:,}** |
                            | Altitude moyenne | {alt_c} m |
                            | Nombre de douars | {nb_d} |
                            """)

                        with col_alerts:
                            if als_c:
                                st.markdown("**🚨 Alertes commune :**")
                                for a in als_c:
                                    css_a = {"ROUGE":"alert-rouge","ORANGE":"alert-orange",
                                             "JAUNE":"alert-jaune"}.get(a[2],"alert-vert")
                                    st.markdown(
                                        f'<div class="{css_a}" style="margin:3px 0;font-size:12px">'
                                        f'<b>{a[0]}</b><br>{a[1]}</div>',
                                        unsafe_allow_html=True)
                            else:
                                st.markdown(
                                    '<div class="alert-vert">✅ Aucune alerte</div>',
                                    unsafe_allow_html=True)

                        # ── Douars de cette commune ──
                        st.markdown(f"**🔵 {nb_d} Douars :**")
                        cols_d = st.columns(4)
                        for i, (_, dr) in enumerate(df_comm.iterrows()):
                            sc_d = compute_score(dr.to_dict(), conditions)
                            lvl_d, em_d, _ = _al(sc_d)
                            with cols_d[i % 4]:
                                st.markdown(
                                    f'<div style="background:rgba(123,184,212,0.06);'
                                    f'border:1px solid rgba(123,184,212,0.18);'
                                    f'border-radius:6px;padding:7px;margin:2px 0;">'
                                    f'{em_d} <b style="font-size:12px">{dr["douar"]}</b><br>'
                                    f'<span style="font-size:11px;opacity:0.8">'
                                    f'👥 {int(dr["population"]):,} hab<br>'
                                    f'⚠️ {int(dr["pop_vulnerable"]):,} vuln.</span></div>',
                                    unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — Top douars critiques
    # ══════════════════════════════════════════════════════════════════════════
    with tab_z3:
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            n_show = st.slider("Nombre de douars", 20, 300, 50)
        with col_d2:
            prov_d = st.selectbox("Province",
                ["Toutes"] + sorted(df_hcp["province_fr"].unique().tolist()), key="z3_prov")
        with col_d3:
            sort_d = st.selectbox("Trier par",
                ["Personnes vulnérables", "Score alerte", "% vulnérables"])

        df_d = df_hcp if prov_d == "Toutes" else df_hcp[df_hcp["province_fr"] == prov_d]
        df_d = df_d.copy()
        df_d["score"]  = df_d.apply(lambda r: compute_score(r.to_dict(), conditions), axis=1)
        df_d["niveau"] = df_d["score"].apply(lambda s: _al(s)[0])

        sort_col = {"Personnes vulnérables": "pop_vulnerable",
                    "Score alerte": "score",
                    "% vulnérables": "pct_vulnerable"}[sort_d]
        top_d = df_d.nlargest(n_show, sort_col).reset_index(drop=True)

        bar_colors = top_d["niveau"].map(
            {"ROUGE":"#D62828","ORANGE":"#F77F00","JAUNE":"#FCBF49","VERT":"#4A7C59"}
        ).fillna("#7BB8D4")

        label_y = (top_d["douar"] + " › " + top_d["commune"]
                   + " › " + top_d["cercle"])[::-1]

        fig_d = go.Figure(go.Bar(
            y=label_y,
            x=top_d["pop_vulnerable"][::-1],
            orientation="h",
            marker_color=list(bar_colors[::-1]),
            text=[f"{v:,} vuln. | Score {s}"
                  for v, s in zip(top_d["pop_vulnerable"][::-1],
                                  top_d["score"][::-1].round(1))],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Vulnérables : %{x:,}<br>"
                "<extra></extra>"
            ),
        ))
        fig_d.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#E8D5B7",
            height=max(500, n_show * 20),
            margin=dict(l=330, r=120, t=30, b=40),
            title=f"Top {n_show} douars · {sort_d} · {prov_d}",
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Personnes vulnérables"),
            yaxis=dict(tickfont_size=10),
        )
        st.plotly_chart(fig_d, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — ANALYSE DÉMOGRAPHIQUE HCP — TOUTE LA RÉGION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Analyse Démographique":
    st.markdown("## 👥 Analyse Démographique — Région Béni Mellal-Khénifra")

    if df_hcp is None or len(df_hcp) == 0:
        st.warning("Aucune donnée HCP chargée. Utilisez le fichier `data/hcp_douar_bnk.csv`.")
        st.stop()

    # ── KPIs région ────────────────────────────────────────────────────────────
    prov_data = aggregate_by_province(df_hcp)
    comm_data = aggregate_by_commune(df_hcp)
    total_pop  = df_hcp["population"].sum()
    total_vuln = df_hcp["pop_vulnerable"].sum()
    total_moins15 = df_hcp["pop_moins_15"].sum()
    total_60plus  = df_hcp["pop_60_plus"].sum()

    st.markdown(f"""
    <div class="header-banner">
        <h3 style="margin:0;color:#7BB8D4;">🗺️ Région Béni Mellal-Khénifra</h3>
        <p style="margin:6px 0;color:#E8D5B7;font-size:0.9rem;">
        5 provinces · {len(prov_data["province_fr"].unique())} provinces ·
        {df_hcp["commune"].nunique()} communes · {len(df_hcp):,} douars
        </p>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("👥 Population totale", f"{total_pop:,}")
    k2.metric("⚠️ Personnes vulnérables", f"{total_vuln:,}", f"{total_vuln/total_pop*100:.1f}%")
    k3.metric("👶 Enfants < 15 ans", f"{total_moins15:,}", f"{total_moins15/total_pop*100:.1f}%")
    k4.metric("🧓 Personnes âgées 60+", f"{total_60plus:,}", f"{total_60plus/total_pop*100:.1f}%")
    k5.metric("🏘 Douars analysés", f"{len(df_hcp):,}")

    st.markdown("---")

    tab_d1, tab_d2, tab_d3, tab_d4, tab_d5, tab_d6 = st.tabs([
        "🗺 Carte Région", "📊 Par Province", "🏘 Top Zones Alertes",
        "📈 Pyramide Âges", "🔍 Explorer Données", "📋 Données brutes"
    ])

    # ── TAB 1 : Carte interactive toute la région ─────────────────────────────
    with tab_d1:
        st.markdown(f"#### 🗺️ Carte des {len(comm_data)} communes — Population vulnérable & Alertes météo")

        view_type = st.radio("Afficher par :", ["Score d'alerte", "Population vulnérable", "% vulnérables", "Altitude"], horizontal=True)

        # Calcul score pour chaque commune
        comm_data["score_alerte"] = comm_data.apply(
            lambda r: compute_vulnerability_score_hcp(r.to_dict(), conditions), axis=1)
        comm_data["niveau_alerte"] = comm_data["score_alerte"].apply(
            lambda s: "🔴 ROUGE" if s >= 60 else "🟠 ORANGE" if s >= 35 else "🟡 JAUNE" if s >= 15 else "🟢 VERT")

        color_col = {"Score d'alerte": "score_alerte",
                     "Population vulnérable": "pop_vulnerable",
                     "% vulnérables": "pct_vulnerable",
                     "Altitude": "altitude"}.get(view_type, "score_alerte")

        color_scale = {"Score d'alerte": "Reds",
                       "Population vulnérable": "YlOrRd",
                       "% vulnérables": "YlOrRd",
                       "Altitude": "Viridis_r"}.get(view_type, "Reds")

        fig_region = px.scatter_mapbox(
            comm_data,
            lat="lat", lon="lon",
            size="pop_vulnerable",
            color=color_col,
            color_continuous_scale=color_scale,
            hover_name="commune",
            hover_data={
                "province_fr": True,
                "cercle": True,
                "population": True,
                "pop_moins_15": True,
                "pop_60_plus": True,
                "pop_vulnerable": True,
                "pct_vulnerable": True,
                "altitude": True,
                "score_alerte": True,
                "niveau_alerte": True,
                "lat": False, "lon": False,
            },
            size_max=35,
            mapbox_style="carto-darkmatter",
            zoom=7.2,
            center={"lat": 32.3, "lon": -6.2},
        )
        fig_region.update_layout(
            height=560,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#E8D5B7",
            coloraxis_colorbar=dict(
                title=dict(text=view_type, font=dict(color="#E8D5B7")),
                tickfont=dict(color="#E8D5B7"),
                bgcolor="rgba(0,0,0,0.5)",
            )
        )
        st.plotly_chart(fig_region, use_container_width=True)

        # Résumé alertes
        n_rouge  = (comm_data["score_alerte"] >= 60).sum()
        n_orange = ((comm_data["score_alerte"] >= 35) & (comm_data["score_alerte"] < 60)).sum()
        n_jaune  = ((comm_data["score_alerte"] >= 15) & (comm_data["score_alerte"] < 35)).sum()
        n_vert   = (comm_data["score_alerte"] < 15).sum()

        ac1, ac2, ac3, ac4 = st.columns(4)
        ac1.metric("🔴 Communes ROUGE",  n_rouge,  "Alerte critique")
        ac2.metric("🟠 Communes ORANGE", n_orange, "Vigilance")
        ac3.metric("🟡 Communes JAUNE",  n_jaune,  "Précaution")
        ac4.metric("🟢 Communes VERT",   n_vert,   "Normal")

        # Explication si tous les scores sont à 0
        if n_rouge == 0 and n_orange == 0 and n_jaune == 0:
            st.info(
                f"ℹ️ **Tous les scores sont à 0** — c'est normal avec {cond_temp}°C.\n\n"
                f"Le score ne s'active qu'à partir de **32°C** (chaleur modérée).\n\n"
                f"**Pour tester des alertes** → montez le curseur 🌡 Température :\n"
                f"- **32°C** → premières alertes JAUNE\n"
                f"- **38°C** → alertes ORANGE\n"
                f"- **42°C** → alertes ROUGE (canicule critique)"
            )

    # ── TAB 2 : Vue par province ───────────────────────────────────────────────
    with tab_d2:
        st.markdown("#### 📊 Profil démographique par province")

        # Barres groupées
        fig_prov = go.Figure()
        province_names = prov_data["province_fr"].tolist()
        colors_list = [PROVINCE_COLORS.get(p, "#7BB8D4") for p in province_names]

        fig_prov.add_trace(go.Bar(
            name="Population totale", x=province_names, y=prov_data["population"],
            marker=dict(color=["#7BB8D4","#C4622D","#4A7C59","#FCBF49","#D62828"],
                        opacity=0.35),
            yaxis="y2",
        ))
        fig_prov.add_trace(go.Bar(
            name="< 15 ans", x=province_names, y=prov_data["pop_moins_15"],
            marker_color="#7BB8D4",
        ))
        fig_prov.add_trace(go.Bar(
            name="60+ ans", x=province_names, y=prov_data["pop_60_plus"],
            marker_color="#C4622D",
        ))
        fig_prov.update_layout(
            barmode="group",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#E8D5B7", height=380,
            margin=dict(l=50, r=60, t=40, b=60),
            title="Population vulnérable vs totale par province",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Personnes vulnérables"),
            yaxis2=dict(overlaying="y", side="right", title="Population totale",
                        showgrid=False, tickfont=dict(color="rgba(232,213,183,0.4)")),
        )
        st.plotly_chart(fig_prov, use_container_width=True)

        # Tableau provinces
        st.markdown("**📋 Synthèse par province :**")
        prov_display = prov_data[["province_fr","population","pop_moins_15","pop_60_plus",
                                    "pop_vulnerable","pct_moins_15","pct_60_plus","pct_vulnerable",
                                    "nb_communes","nb_douars"]].copy()
        prov_display.columns = ["Province","Population","< 15 ans","60+ ans","Vulnérables",
                                  "% < 15","% 60+","% Vulnérable","Communes","Douars"]
        st.dataframe(prov_display, use_container_width=True, hide_index=True)

        # Carte radar par province
        st.markdown("#### 🕸 Profil de vulnérabilité par province")
        fig_rad = go.Figure()
        cats = ["% Enfants < 15", "% Âgés 60+", "% Vulnérables", "Nb douars (norm.)", "Pop. (norm.)"]
        max_douars = prov_data["nb_douars"].max()
        max_pop    = prov_data["population"].max()
        prov_colors = ["#7BB8D4","#C4622D","#4A7C59","#FCBF49","#D62828"]
        for i, (_, p) in enumerate(prov_data.iterrows()):
            vals = [
                float(p["pct_moins_15"]) * 2,
                float(p["pct_60_plus"]) * 5,
                float(p["pct_vulnerable"]) * 3,
                p["nb_douars"] / max_douars * 100,
                p["population"] / max_pop * 100,
            ]
            vals = [min(v, 100) for v in vals]
            fig_rad.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                fill="toself", name=p["province_fr"],
                line_color=prov_colors[i], opacity=0.7,
            ))
        fig_rad.update_layout(
            polar=dict(bgcolor="rgba(0,0,0,0)",
                       radialaxis=dict(visible=True, range=[0, 100],
                                       gridcolor="rgba(255,255,255,0.1)",
                                       tickfont=dict(color="#7BB8D4"))),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#E8D5B7", height=420,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=40, r=40, t=40, b=40),
        )
        st.plotly_chart(fig_rad, use_container_width=True)

    # ── TAB 3 : Top zones alertes ─────────────────────────────────────────────
    with tab_d3:
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            n_top = st.slider("Nombre de zones", 10, 119, 30)
        with col_f2:
            prov_filter = st.selectbox("Filtrer province",
                ["Toutes"] + sorted(df_hcp["province_fr"].unique().tolist()))
        with col_f3:
            sort_by = st.selectbox("Trier par", ["Pop. vulnérable", "Score alerte", "% vulnérables"])

        df_filter = df_hcp if prov_filter == "Toutes" else df_hcp[df_hcp["province_fr"] == prov_filter]
        top_comm = aggregate_by_commune(df_filter)
        top_comm["score_alerte"] = top_comm.apply(
            lambda r: compute_vulnerability_score_hcp(r.to_dict(), conditions), axis=1)

        sort_col_map = {"Pop. vulnérable": "pop_vulnerable",
                        "Score alerte": "score_alerte",
                        "% vulnérables": "pct_vulnerable"}
        top_comm = top_comm.nlargest(n_top, sort_col_map[sort_by]).reset_index(drop=True)

        # Couleur par niveau d'alerte
        def get_alert_color(sc):
            if sc >= 60:  return "#D62828"
            elif sc >= 35: return "#F77F00"
            elif sc >= 15: return "#FCBF49"
            else:          return "#4A7C59"

        bar_colors = [get_alert_color(s) for s in top_comm["score_alerte"]]

        fig_top = go.Figure(go.Bar(
            y=top_comm["commune"][::-1] + " (" + top_comm["province_fr"][::-1] + ")",
            x=top_comm["pop_vulnerable"][::-1],
            orientation="h",
            marker_color=bar_colors[::-1],
            text=[f"{v:,} vuln. | Score {s}" for v, s in
                  zip(top_comm["pop_vulnerable"][::-1], top_comm["score_alerte"][::-1])],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Pop. vulnérable: %{x:,}<br>"
                "Population: %{customdata[0]:,}<br>"
                "Altitude: %{customdata[1]}m<br>"
                "<extra></extra>"
            ),
            customdata=list(zip(top_comm["population"][::-1], top_comm["altitude"][::-1].astype(int))),
        ))
        fig_top.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#E8D5B7", height=max(450, n_top * 22),
            margin=dict(l=240, r=100, t=40, b=40),
            title=f"Top {n_top} communes — {sort_by} | {prov_filter}",
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Personnes vulnérables"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont_size=11),
        )
        st.plotly_chart(fig_top, use_container_width=True)

        # Fiches alertes top 10
        st.markdown("**🚨 Fiches d'alerte — Top 10 zones prioritaires :**")
        top10 = top_comm.head(10)
        cols_alert = st.columns(2)
        for i, (_, row) in enumerate(top10.iterrows()):
            sc = row["score_alerte"]
            lvl, em, css = alert_level(sc)
            with cols_alert[i % 2]:
                pct_v = row.get("pct_vulnerable", 0)
                alt   = int(row.get("altitude", 0))
                st.markdown(f"""
                <div class="{css}" style="margin-bottom:6px">
                <b>{em} {row['commune']}</b>
                <span style="opacity:0.7;font-size:11px"> — {row['province_fr']}</span><br>
                <small>
                👥 Pop: <b>{row['population']:,}</b> |
                ⚠️ Vulnérables: <b>{row['pop_vulnerable']:,}</b> ({pct_v:.1f}%)<br>
                👶 Enfants: {row['pop_moins_15']:,} |
                🧓 Âgés: {row['pop_60_plus']:,} |
                ⛰️ Alt: {alt}m<br>
                Score: <b>{sc}/100</b>
                </small>
                </div>
                """, unsafe_allow_html=True)

    # ── TAB 4 : Pyramide des âges ──────────────────────────────────────────────
    with tab_d4:
        st.markdown("#### 📈 Structure par âge — Région Béni Mellal-Khénifra")

        # Pyramide globale
        labels = ["< 15 ans", "15 – 59 ans", "60+ ans"]
        values = [total_moins15, df_hcp["pop_15_59"].sum(), total_60plus]
        colors_pie = ["#7BB8D4", "#4A7C59", "#C4622D"]

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            fig_pie = go.Figure(go.Pie(
                labels=labels, values=values,
                marker_colors=colors_pie,
                hole=0.45,
                textinfo="label+percent+value",
                textfont_size=12,
            ))
            fig_pie.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#E8D5B7", height=350,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False,
                title="Région — Structure d'âge globale",
                annotations=[dict(text=f"{total_pop:,}<br>hab.", x=0.5, y=0.5,
                                   font_size=13, showarrow=False, font_color="#E8D5B7")]
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_p2:
            # Comparaison % vulnérables par province
            fig_pct = go.Figure()
            prov_sorted = prov_data.sort_values("pct_vulnerable", ascending=True)
            fig_pct.add_trace(go.Bar(
                y=prov_sorted["province_fr"],
                x=prov_sorted["pct_moins_15"],
                orientation="h", name="% < 15 ans",
                marker_color="#7BB8D4",
            ))
            fig_pct.add_trace(go.Bar(
                y=prov_sorted["province_fr"],
                x=prov_sorted["pct_60_plus"],
                orientation="h", name="% 60+ ans",
                marker_color="#C4622D",
            ))
            fig_pct.update_layout(
                barmode="stack",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#E8D5B7", height=350,
                margin=dict(l=120, r=20, t=40, b=40),
                title="% Population vulnérable par province",
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="%"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            )
            st.plotly_chart(fig_pct, use_container_width=True)

        # Distribution % vulnérables par commune (histogramme)
        st.markdown("#### Distribution — % Personnes vulnérables dans les 119 communes")
        fig_hist = go.Figure(go.Histogram(
            x=comm_data["pct_vulnerable"].dropna(),
            nbinsx=25,
            marker_color="#7BB8D4",
            marker_line_color="#0D1B2A",
            marker_line_width=1,
        ))
        fig_hist.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#E8D5B7", height=280,
            margin=dict(l=50, r=20, t=30, b=50),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="% Population vulnérable"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Nombre de communes"),
        )
        med_vuln = comm_data["pct_vulnerable"].median()
        fig_hist.add_vline(x=med_vuln, line_dash="dash", line_color="#C4622D",
                           annotation_text=f"Médiane: {med_vuln:.1f}%", annotation_font_color="#E8D5B7")
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── TAB 5 : Explorer ──────────────────────────────────────────────────────
    with tab_d5:
        st.markdown("#### 🔍 Explorer les données par commune")

        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            prov_sel = st.selectbox("Province", ["Toutes"] + sorted(df_hcp["province_fr"].unique().tolist()))
        with col_e2:
            cercles = ["Tous"] + sorted(df_hcp["cercle"].unique().tolist()) if prov_sel == "Toutes" \
                      else ["Tous"] + sorted(df_hcp[df_hcp["province_fr"]==prov_sel]["cercle"].unique().tolist())
            cercle_sel = st.selectbox("Cercle", cercles)
        with col_e3:
            search_q = st.text_input("🔍 Commune / douar (arabe ou latin)")

        df_explore = df_hcp.copy()
        if prov_sel != "Toutes":
            df_explore = df_explore[df_explore["province_fr"] == prov_sel]
        if cercle_sel != "Tous":
            df_explore = df_explore[df_explore["cercle"] == cercle_sel]
        if search_q:
            df_explore = df_explore[
                df_explore["commune"].str.contains(search_q, na=False) |
                df_explore["douar"].str.contains(search_q, na=False)
            ]

        # Agrégation commune pour la sélection
        comm_sel = aggregate_by_commune(df_explore)
        comm_sel["score_alerte"] = comm_sel.apply(
            lambda r: compute_vulnerability_score_hcp(r.to_dict(), conditions), axis=1)

        st.markdown(f"**{len(comm_sel)} communes | {df_explore['population'].sum():,} hab. | {df_explore['pop_vulnerable'].sum():,} vulnérables**")

        if len(comm_sel) > 0:
            # Mini carte de la sélection
            fig_sel = px.scatter_mapbox(
                comm_sel, lat="lat", lon="lon",
                size="pop_vulnerable", color="score_alerte",
                color_continuous_scale="RdYlGn_r",
                hover_name="commune",
                hover_data={"province_fr": True, "population": True,
                             "pop_vulnerable": True, "pct_vulnerable": True,
                             "altitude": True, "lat": False, "lon": False},
                size_max=30,
                mapbox_style="carto-darkmatter",
                zoom=8.5,
                center={"lat": comm_sel["lat"].mean(), "lon": comm_sel["lon"].mean()},
            )
            fig_sel.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0),
                                   paper_bgcolor="rgba(0,0,0,0)", font_color="#E8D5B7")
            st.plotly_chart(fig_sel, use_container_width=True)

        # Tableau détaillé douars
        cols_show = ["province_fr","cercle","commune","douar","population",
                     "pct_moins_15","pct_60_plus","pop_vulnerable","pct_vulnerable","altitude"]
        cols_show = [c for c in cols_show if c in df_explore.columns]
        st.dataframe(df_explore[cols_show].reset_index(drop=True), use_container_width=True)

    # ── TAB 6 : Données brutes ────────────────────────────────────────────────
    with tab_d6:
        st.markdown(f"**{len(df_hcp):,} douars | {df_hcp['population'].sum():,} habitants | région complète**")

        cols_dl = ["province_fr","cercle","commune","douar","population",
                   "pct_moins_15","pct_15_59","pct_60_plus",
                   "pop_moins_15","pop_15_59","pop_60_plus",
                   "pop_vulnerable","pct_vulnerable","lat","lon","altitude"]
        cols_dl = [c for c in cols_dl if c in df_hcp.columns]

        st.dataframe(df_hcp[cols_dl].head(200), use_container_width=True)
        st.download_button("⬇️ Télécharger CSV complet (toute la région)",
                           df_hcp[cols_dl].to_csv(index=False),
                           "hcp_region_bm_khenifra_enrichi.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — EXPORT QGIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🌍 Export QGIS":
    st.markdown("## 🌍 Intégration QGIS — Export & Scripts")

    tab_q1, tab_q2, tab_q3, tab_q4 = st.tabs([
        "📤 Export GeoJSON/CSV", "🐍 Script PyQGIS", "📋 Script WFS", "📖 Guide QGIS"
    ])

    with tab_q1:
        st.markdown("#### Export Zones de Vulnérabilité — GeoJSON (EPSG:4326)")

        # GeoJSON zones d'alerte
        geojson = zones_to_geojson(df_zones, conditions)
        geojson_str = json.dumps(geojson, ensure_ascii=False, indent=2)
        st.code(geojson_str[:2000] + "\n...", language="json")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("⬇️ GeoJSON Alertes", geojson_str,
                               "beni_mellal_alertes.geojson", "application/json")
        with c2:
            # CSV zones
            df_exp = df_zones.copy()
            for _, z in df_zones.iterrows():
                sc, vp, _ = compute_vulnerability_score(z.to_dict(), conditions)
                lvl, _, _ = alert_level(sc)
                df_exp.loc[df_exp["id"] == z["id"], "score_vulnerabilite"] = sc
                df_exp.loc[df_exp["id"] == z["id"], "pop_vulnerable"] = vp
                df_exp.loc[df_exp["id"] == z["id"], "niveau_alerte"] = lvl
            st.download_button("⬇️ CSV Zones (lat/lon)", df_exp.to_csv(index=False),
                               "beni_mellal_zones.csv", "text/csv")
        with c3:
            monthly_agg = get_monthly_stats(df_hist)
            st.download_button("⬇️ Historique mensuel CSV", monthly_agg.to_csv(index=False),
                               "historique_mensuel.csv", "text/csv")

        # Export HCP GeoJSON — toute la région
        if df_hcp is not None and len(df_hcp) > 0:
            st.markdown("#### Export Région Complète — 119 communes, GeoJSON")
            hcp_geojson = json.dumps(build_region_geojson(df_hcp, conditions), ensure_ascii=False, indent=2)
            st.code(hcp_geojson[:1500] + "\n...", language="json")
            st.download_button("⬇️ GeoJSON Région Béni Mellal-Khénifra (119 communes)",
                               hcp_geojson, "bm_khenifra_region_alertes.geojson", "application/json")

    with tab_q2:
        pyqgis_script = '''# ── Script PyQGIS — BeniMeteo Béni Mellal-Khénifra ─────────────────────────
# Copiez dans la Console Python QGIS (Plugins > Console Python > Éditeur)
# Modifiez les chemins ci-dessous avant d'exécuter.

from qgis.core import (
    QgsProject, QgsVectorLayer, QgsSymbol,
    QgsCategorizedSymbolRenderer, QgsRendererCategory,
    QgsSimpleMarkerSymbolLayer, QgsProperty
)
from qgis.PyQt.QtGui import QColor
import os, json

# ── 1. Chemins vers les fichiers exportés depuis BeniMeteo ───────────────────
BASE_DIR         = r"C:/BeniMeteo/exports"   # ← MODIFIEZ CE CHEMIN
GEOJSON_ALERTES  = os.path.join(BASE_DIR, "beni_mellal_alertes.geojson")
GEOJSON_HCP      = os.path.join(BASE_DIR, "hcp_communes_alertes.geojson")
CSV_HISTORIQUE   = os.path.join(BASE_DIR, "historique_mensuel.csv")

def charger_couche(path, nom):
    layer = QgsVectorLayer(path, nom, "ogr")
    if not layer.isValid():
        print(f"❌ Couche invalide : {path}")
        return None
    print(f"✅ Chargée : {nom} ({layer.featureCount()} entités)")
    return layer

def appliquer_symbologie_alerte(layer, champ="niveau_alerte"):
    """Symbologie catégorisée par niveau d'alerte."""
    palettes = {
        "ROUGE":  (QColor(214, 40,  40),  12, "🔴 Alerte Rouge"),
        "ORANGE": (QColor(247, 127,  0),  10, "🟠 Alerte Orange"),
        "JAUNE":  (QColor(252, 191, 73),   8, "🟡 Alerte Jaune"),
        "VERT":   (QColor(74,  124, 89),   6, "🟢 Aucune alerte"),
    }
    categories = []
    for val, (color, size, label) in palettes.items():
        symbol = QgsSymbol.defaultSymbol(layer.geometryType())
        symbol.setColor(color)
        if hasattr(symbol, 'setSize'):
            symbol.setSize(size)
        categories.append(QgsRendererCategory(val, symbol, label))
    layer.setRenderer(QgsCategorizedSymbolRenderer(champ, categories))
    layer.triggerRepaint()

# ── 2. Charger couche alertes zones ─────────────────────────────────────────
layer_alertes = charger_couche(GEOJSON_ALERTES, "BeniMeteo_Alertes_Zones")
if layer_alertes:
    appliquer_symbologie_alerte(layer_alertes)
    QgsProject.instance().addMapLayer(layer_alertes)

# ── 3. Charger couche HCP communes ──────────────────────────────────────────
layer_hcp = charger_couche(GEOJSON_HCP, "BeniMeteo_HCP_Communes")
if layer_hcp:
    appliquer_symbologie_alerte(layer_hcp)
    QgsProject.instance().addMapLayer(layer_hcp)

# ── 4. Fond de carte OpenStreetMap ───────────────────────────────────────────
xyz_url = "type=xyz&url=https://tile.openstreetmap.org/{z}/{x}/{y}.png&zmax=19&zmin=0"
osm = QgsVectorLayer(xyz_url, "OpenStreetMap", "wms")
# Pour OSM : utilisez le gestionnaire de couches XYZ de QGIS

# ── 5. Centrer la vue sur Béni Mellal ────────────────────────────────────────
iface.mapCanvas().setCenter(
    QgsProject.instance().transformContext()
)

print("\\n✅ BeniMeteo chargé dans QGIS !")
print("   Couches : Alertes zones + HCP communes")
print("   Projection : EPSG:4326 (WGS84)")
print("   → Convertir en EPSG:26191 (Lambert Maroc) si besoin")
'''
        st.code(pyqgis_script, language="python")
        st.download_button("⬇️ Télécharger script PyQGIS", pyqgis_script,
                           "beni_mellal_qgis.py", "text/plain")

    with tab_q3:
        wfs_script = '''# ── Script connexion WFS / GeoServer ────────────────────────────────────────
# Pour intégration BeniMeteo → GeoServer → QGIS

# 1. Dans GeoServer : publier le GeoJSON comme couche WFS
#    URL de publication : http://votre-geoserver:8080/geoserver/beniMeteo/wfs

# 2. Dans QGIS, ajouter une couche WFS :
from qgis.core import QgsVectorLayer, QgsProject

wfs_url = (
    "http://localhost:8080/geoserver/beniMeteo/wfs?"
    "service=WFS&version=2.0.0&request=GetFeature"
    "&typeName=beniMeteo:alertes_zones&outputFormat=application/json"
)

layer_wfs = QgsVectorLayer(f"url='{wfs_url}' typename='alertes_zones'", "BeniMeteo WFS", "WFS")
if layer_wfs.isValid():
    QgsProject.instance().addMapLayer(layer_wfs)
    print("✅ Couche WFS connectée")
else:
    print("❌ Vérifiez l'URL GeoServer")

# 3. Mise à jour automatique (rafraîchissement)
layer_wfs.setAutoRefreshInterval(300000)  # 5 minutes
layer_wfs.setAutoRefreshEnabled(True)

# 4. Connexion PostGIS pour stockage persistant
from qgis.core import QgsDataSourceUri
uri = QgsDataSourceUri()
uri.setConnection("localhost", "5432", "beni_mellal_climate", "user", "password")
uri.setDataSource("public", "alertes_zones", "geom", "", "id")
layer_pg = QgsVectorLayer(uri.uri(False), "BeniMeteo PostGIS", "postgres")
'''
        st.code(wfs_script, language="python")

    with tab_q4:
        st.markdown("""
        #### 📖 Guide Intégration QGIS — Béni Mellal-Khénifra

        **Étape 1 — Exporter depuis BeniMeteo**
        - Onglet *Export GeoJSON/CSV* → téléchargez `beni_mellal_alertes.geojson` et `hcp_communes_alertes.geojson`

        **Étape 2 — Importer dans QGIS**
        - `Couche > Ajouter une couche vecteur` → sélectionnez le `.geojson`
        - Projection : **EPSG:4326** (WGS84) — source
        - Optionnel : convertir en **EPSG:26191** (Lambert Maroc Nord) pour les analyses métriques

        **Étape 3 — Symbologie par niveau d'alerte**
        - Clic droit → `Propriétés > Symbologie > Catégorisé`
        - Champ : `niveau_alerte`
        - Couleurs : ROUGE=#D62828, ORANGE=#F77F00, JAUNE=#FCBF49, VERT=#4A7C59

        **Étape 4 — Couches de contexte recommandées**
        - 🏔 MNT Atlas 30m (NASA SRTM) — risques pentes et altitude
        - 🗺 OSM Béni Mellal — réseau routier, bâtiments
        - 🌊 Hydrologie ABHOER — oueds et zones inondables
        - 📍 Limites administratives Maroc (HCP shapefile)

        **Étape 5 — Automatisation**
        - Script PyQGIS fourni dans l'onglet précédent
        - Planification avec **QGIS Task Manager** plugin
        - Connexion GeoServer pour WMS/WFS temps réel

        **Projections recommandées :**
        - `EPSG:4326` — WGS84 (import/export, API)
        - `EPSG:26191` — Lambert Conique Conforme Maroc Nord (mesures, analyses)
        - `EPSG:3857` — Web Mercator (fonds de carte OSM/Google)
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — SAISIE & IMPORT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Saisie & Import":
    st.markdown("## ⚙️ Saisie Manuelle & Import de Données")

    tab_s1, tab_s2, tab_s3 = st.tabs(["✍️ Saisie météo", "📁 Import fichier", "👥 Données démographiques"])

    with tab_s1:
        st.markdown("#### Ajouter une mesure journalière")
        with st.form("saisie_meteo_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                s_date   = st.date_input("📅 Date", datetime.today())
                s_tmax   = st.number_input("🌡 T° Max (°C)", -10.0, 55.0, 30.0, step=0.1)
                s_tmin   = st.number_input("❄ T° Min (°C)", -10.0, 50.0, 18.0, step=0.1)
                s_precip = st.number_input("🌧 Précipitations (mm)", 0.0, 200.0, 0.0, step=0.1)
            with c2:
                s_hum   = st.number_input("💧 Humidité (%) max", 0.0, 100.0, 60.0)
                s_wind  = st.number_input("💨 Vent Max (km/h)", 0.0, 150.0, 15.0)
                s_wind_d = st.selectbox("🧭 Direction Vent", ["N","NE","E","SE","S","SW","W","NW"])
                s_press  = st.number_input("📊 Pression (hPa)", 950.0, 1050.0, 1013.0)
            with c3:
                s_uv    = st.slider("☀ UV Max", 0, 11, 5)
                s_sun   = st.number_input("🌞 Ensoleillement (h)", 0.0, 14.0, 8.0, step=0.1)
                s_snow  = st.number_input("❄ Neige (mm)", 0.0, 50.0, 0.0)
                s_notes = st.text_area("📝 Notes", placeholder="Observations, événements exceptionnels…")

            submitted = st.form_submit_button("💾 Enregistrer la mesure")
            if submitted:
                t_mean = round((s_tmax + s_tmin) / 2, 1)
                hi = max(0, round(
                    -8.78 + 1.61*t_mean + 2.34*s_hum - 0.146*t_mean*s_hum
                    - 0.0123*t_mean**2 - 0.0164*s_hum**2, 1))
                st.success(f"✅ Mesure du **{s_date}** — T° moy: {t_mean}°C | Indice chaleur: {hi}°C | Précip: {s_precip}mm")
                conds_mesure = dict(temp=t_mean, precip=s_precip, wind=s_wind, humidity=s_hum, uv=s_uv)
                st.markdown("**Alertes calculées pour cette mesure :**")
                for _, z in df_zones.iloc[:3].iterrows():
                    sc, vp, als = compute_vulnerability_score(z.to_dict(), conds_mesure)
                    lvl, em, css = alert_level(sc)
                    if als:
                        st.markdown(
                            f'<div class="{css}"><b>{z["nom"]}</b> {em} {lvl} — '
                            f'{als[0][0]}: {als[0][1]}</div>',
                            unsafe_allow_html=True)

                new_row = pd.DataFrame([{
                    "date": pd.to_datetime(s_date), "tmax": s_tmax, "tmin": s_tmin,
                    "precip": s_precip, "humidity_max": s_hum, "wind_max": s_wind,
                    "uv_max": s_uv, "pressure": s_press, "snowfall": s_snow,
                }])
                st.download_button("⬇️ Télécharger cette mesure", new_row.to_csv(index=False),
                                   f"mesure_{s_date}.csv", "text/csv")

    with tab_s2:
        st.markdown("#### Import depuis CSV ou Excel")
        st.markdown("""
        <div class="info-box">
        <b>Format CSV attendu :</b><br>
        <code>date,tmax,tmin,precip,humidity_max,wind_max,pressure,uv_max</code><br>
        <code>2024-06-01,38.2,22.1,0,35,18,1010,9</code>
        </div>
        """, unsafe_allow_html=True)
        uploaded = st.file_uploader("Choisir un fichier", type=["csv", "xlsx", "xls"])
        if uploaded:
            try:
                df_up = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
                st.success(f"✅ {len(df_up):,} lignes | colonnes: {list(df_up.columns)}")
                st.dataframe(df_up.head(30), use_container_width=True)
                num_cols = df_up.select_dtypes(include=[np.number]).columns.tolist()
                if num_cols:
                    st.markdown("**Statistiques :**")
                    st.dataframe(df_up[num_cols].describe().round(2), use_container_width=True)
            except Exception as e:
                st.error(f"❌ Erreur d'import : {e}")

    with tab_s3:
        st.markdown("#### Aperçu des données démographiques HCP")
        st.markdown("""
        <div class="info-box">
        <b>Fichier chargé :</b> <code>data/hcp_douar_bnk.csv</code><br>
        Colonnes : <code>code_geo; province; cercle; commune; douar; population; age_m15; age_15_59; age_60_plus</code><br>
        Pour mettre à jour : chargez un nouveau CSV via la barre latérale.
        </div>
        """, unsafe_allow_html=True)
        if df_hcp is not None:
            st.markdown(f"**{len(df_hcp):,} douars chargés | {df_hcp['population'].sum():,} habitants total**")
            prov_summary = df_hcp.groupby("province")["population"].sum().reset_index()
            prov_summary.columns = ["Province", "Population"]
            st.dataframe(prov_summary, use_container_width=True, hide_index=True)
        template = pd.DataFrame([
            {"province":"بني ملال","cercle":"بني ملال","commune":"مدينة بني ملال",
             "douar":"المركز","population":5000,"age_m15":28.5,"age_15_59":59.0,"age_60_plus":12.5}
        ])
        st.download_button("⬇️ Template CSV HCP", template.to_csv(index=False, sep=";"),
                           "template_hcp.csv", "text/csv")



# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — NOTIFICATIONS EMAIL
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📧 Notifications":
    st.markdown("## 📧 Configuration des Notifications")

    cfg = load_notif_config()

    # Guide Gmail bien visible
    st.markdown("""
    <div class="alert-orange">
    <b>⚠️ Mot de passe d'APPLICATION requis — pas votre vrai mot de passe Gmail</b><br>
    <ol style="margin:8px 0 4px 20px;padding:0">
      <li>Allez sur <b>myaccount.google.com</b></li>
      <li>Cliquez sur <b>Sécurité</b></li>
      <li>Activez la <b>Vérification en 2 étapes</b> si ce n'est pas fait</li>
      <li>Cherchez <b>Mots de passe des applications</b> → cliquez dessus</li>
      <li>Choisissez <b>Autre</b>, tapez <b>BeniMeteo</b>, cliquez <b>Générer</b></li>
      <li>Copiez le code de <b>16 lettres</b> (ex : <code>abcd efgh ijkl mnop</code>)</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Paramètres")
    col_n1, col_n2 = st.columns(2)
    with col_n1:
        email_en  = st.toggle("Activer les notifications email", value=cfg.get("email_enabled", False))
        sender    = st.text_input("Adresse Gmail expéditrice", value=cfg.get("email_sender", ""),
                                   placeholder="votre.email@gmail.com")
        password  = st.text_input("Mot de passe d'APPLICATION (16 caractères)",
                                   type="password", value=cfg.get("email_password", ""),
                                   placeholder="abcdefghijklmnop",
                                   help="Ce n'est PAS votre vrai mot de passe Gmail. Voir guide ci-dessus.")
    with col_n2:
        recipients_str = st.text_area("Destinataires (un par ligne)",
                                       value="\n".join(cfg.get("email_recipients", [])),
                                       placeholder="exemple@gmail.com\nautre@outlook.com")
        # Validation en temps réel des adresses
        if recipients_str.strip():
            recip_list = [r.strip() for r in recipients_str.strip().split("\n") if r.strip()]
            invalids = [r for r in recip_list if "@" not in r or "." not in r.split("@")[-1]]
            if invalids:
                st.error(f"❌ Adresse(s) incorrecte(s) : {', '.join(invalids)}")
            else:
                st.caption(f"✅ {len(recip_list)} destinataire(s) valide(s)")
        seuil_notif = st.selectbox("Seuil de notification",
            ["ROUGE", "ORANGE", "JAUNE"],
            index=["ROUGE","ORANGE","JAUNE"].index(cfg.get("seuil_notification","ORANGE")))
        heure_notif = st.text_input("Heure d'envoi quotidien (HH:MM)",
                                     value=cfg.get("notification_heure","07:00"))

    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        if st.button("💾 Sauvegarder"):
            # Supprimer les espaces du mot de passe d'application (Gmail les affiche avec espaces)
            clean_password = password.replace(" ", "").strip()
            new_cfg = {
                "email_enabled":    email_en,
                "smtp_server":      "smtp.gmail.com",
                "smtp_port":        587,
                "email_sender":     sender.strip(),
                "email_password":   clean_password,
                "email_recipients": [r.strip() for r in recipients_str.strip().split("\n") if r.strip()],
                "seuil_notification": seuil_notif,
                "notification_heure": heure_notif,
            }
            save_notif_config(new_cfg)
            if len(clean_password) == 16:
                st.success("✅ Configuration sauvegardée — mot de passe 16 caractères OK")
            elif len(clean_password) > 0:
                st.warning(f"⚠️ Mot de passe = {len(clean_password)} caractères (attendu : 16 sans espaces)")

    with col_b2:
        if st.button("🧪 Tester connexion"):
            test_cfg = {**cfg, "email_sender": sender, "email_password": password}
            res = test_email_connection(test_cfg)
            if res["success"]:
                st.success(f"✅ {res['message']}")
            else:
                st.error(f"❌ {res['message']}")

    with col_b3:
        if st.button("📨 Envoyer email test"):
            from src.notifications import send_alert_email
            from src.alert_engine import get_forecast_conditions, compute_score_v2, alert_level_v2
            from src.demographics import aggregate_by_commune

            # Utiliser les vraies données météo + vraies communes
            with st.spinner("Calcul des vraies alertes..."):
                forecast_reel = get_forecast_conditions(lat_input, lon_input, days=1)
                comm_agg = aggregate_by_commune(df_hcp)
                seasonal_now = get_seasonal_context()

                if forecast_reel:
                    fc_today = forecast_reel[0]
                    # Ajouter le mois pour le calcul des seuils hybrides DMN
                    from datetime import date as _dt
                    fc_today_complet = {**fc_today,
                        "mois": int(fc_today.get("date","2026-01-01")[5:7])
                                  if fc_today.get("date") else _dt.today().month,
                        "tmax": fc_today.get("tmax", fc_today.get("temp", 25)),
                    }
                    tmax_today = fc_today_complet.get("tmax", 25)
                    rows_reel = []
                    for _, comm in comm_agg.iterrows():
                        sc = compute_score_v2(comm.to_dict(), fc_today_complet,
                                              seasonal_context=seasonal_now)
                        lvl, em, _ = alert_level_v2(sc)
                        if sc >= 12:  # inclure JAUNE et plus
                            rows_reel.append({
                                "commune": comm.get("commune",""),
                                "province_fr": comm.get("province_fr",""),
                                "score": sc,
                                "niveau": lvl,
                                "pop_vulnerable": int(comm.get("pop_vulnerable",0)),
                                "population": int(comm.get("population",0)),
                                "tmax": tmax_today,
                            })
                    if rows_reel:
                        test_df = pd.DataFrame(rows_reel).sort_values("score", ascending=False).head(10)
                    else:
                        # Pas d'alerte active = email informatif VERT
                        from src.alert_engine import NORMALES_DMN, EXCES_PAR_MOIS
                        mois_a = fc_today_complet.get("mois", _dt.today().month)
                        norm_a = NORMALES_DMN.get(mois_a,(27.5,0,0))[0]
                        ej_a, eo_a, er_a = EXCES_PAR_MOIS.get(mois_a,(4,6,8))
                        exces_a = round(tmax_today - norm_a, 1)
                        test_df = pd.DataFrame([{
                            "commune": f"Conditions normales — Tmax={tmax_today}°C (normale={norm_a}°C, excès={exces_a:+.1f}°C)",
                            "province_fr": "Région Béni Mellal-Khénifra",
                            "score": 0,
                            "niveau": "VERT",
                            "pop_vulnerable": int(comm_agg["pop_vulnerable"].sum()),
                            "population": int(comm_agg["population"].sum()),
                            "tmax": tmax_today,
                        }])
                        st.info(
                            f"✅ Aucune anomalie thermique — Tmax={tmax_today}°C vs normale "
                            f"{norm_a}°C (excès {exces_a:+.1f}°C). "
                            f"Seuil alerte JAUNE = {norm_a+ej_a}°C. "
                            f"Bulletin VERT envoyé : conditions normales pour la saison."
                        )
                    snapshot_reel = {
                        "nb_rouge":  int((test_df["niveau"]=="ROUGE").sum()),
                        "nb_orange": int((test_df["niveau"]=="ORANGE").sum()),
                        "nb_jaune":  int((test_df["niveau"]=="JAUNE").sum()),
                        "nb_vert":   int((test_df["niveau"]=="VERT").sum()),
                        "tmax_region": tmax_today,
                    }
                    st.info(f"📊 Email basé sur prévisions réelles — Tmax={tmax_today}°C | "
                            f"{snapshot_reel['nb_rouge']} communes ROUGE | "
                            f"{snapshot_reel['nb_orange']} ORANGE")
                else:
                    # Pas internet — utiliser simulation
                    test_df = pd.DataFrame([{
                        "commune": "Simulation — pas de connexion",
                        "province_fr": "Béni Mellal",
                        "score": 0, "niveau": "VERT",
                        "pop_vulnerable": 0, "population": 0, "tmax": 25,
                    }])
                    snapshot_reel = {}
                    st.warning("⚠️ Open-Meteo inaccessible — email de test sans données réelles")

            test_cfg = {**cfg, "email_sender": sender, "email_password": password,
                        "email_recipients": [r.strip() for r in recipients_str.strip().split("\n") if r.strip()],
                        "email_enabled": True}
            res = send_alert_email(test_df, snapshot_reel, test_cfg)
            if res["success"]:
                st.success(f"✅ {res['message']}")
            else:
                for line in res['message'].split('\n'):
                    if line.strip():
                        st.error(line) if 'incorrect' in line.lower() or 'erreur' in line.lower() \
                            else st.warning(line)

    st.markdown("---")
    st.markdown("### 🔍 Diagnostic — Pourquoi je ne reçois pas l'email ?")

    with st.expander("Afficher le diagnostic complet", expanded=True):
        st.markdown("""
        **Vérifiez dans cet ordre :**

        **1. Dossier Spams / Indésirables**
        L'email envoyé depuis un script Python arrive presque toujours dans les spams.
        Regardez le dossier **Spam / Indésirables** de l'adresse destinataire.

        **2. L'email expéditeur = l'email destinataire ?**
        Gmail bloque parfois les emails envoyés à soi-même via SMTP.
        Essayez avec **une adresse destinataire différente** (ex: Outlook, Yahoo).

        **3. Vérifier le journal des envois**
        """)

        log_path = os.path.join("data", "notifications_log.csv")
        if os.path.exists(log_path):
            try:
                import csv
                with open(log_path, "r", encoding="utf-8") as f:
                    lines_log = f.readlines()
                st.code("".join(lines_log[-10:]), language="text")
            except:
                st.caption("Journal vide ou inaccessible")
        else:
            st.caption("Aucun envoi enregistré pour l'instant.")

        st.markdown("""
        **4. Test manuel depuis Python**
        Si le bouton affiche ✅ mais rien ne reçoit, ouvrez un terminal dans le dossier BeniMeteo et tapez :
        """)
        st.code("""python -c "
import smtplib
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login('VOTRE_EMAIL@gmail.com', 'MOT_DE_PASSE_APPLICATION')
server.sendmail('VOTRE_EMAIL@gmail.com', 'DESTINATAIRE@gmail.com',
    'Subject: Test BeniMeteo\\n\\nCeci est un test.')
server.quit()
print('Email envoyé !')
"
        """, language="bash")

        st.markdown("""
        **5. Activer l'accès SMTP dans Gmail**
        Si vous avez un compte Google Workspace (université), l'administrateur
        doit peut-être autoriser l'accès SMTP externe.
        Pour Gmail personnel : myaccount.google.com → Sécurité →
        vérifiez que la **Vérification en 2 étapes est activée** avant de créer
        le mot de passe d'application.

        **6. Attendre 2-3 minutes**
        Gmail peut avoir un délai avant la livraison.
        """)

    st.markdown("---")
    st.markdown("### 🕐 Planification automatique")
    st.info("Le scheduler envoie automatiquement les alertes chaque matin à l'heure configurée.")
    logs = get_scheduler_logs(20)
    if len(logs) > 0:
        st.dataframe(logs, use_container_width=True, hide_index=True)
    else:
        st.caption("Aucun log scheduler pour l'instant.")





# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — RAPPORTS PDF
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📄 Rapports PDF":
    st.markdown("## 📄 Génération de Rapports PDF")

    tab_p1, tab_p2 = st.tabs(["📝 Générer un rapport", "📁 Rapports existants"])

    with tab_p1:
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            report_type = st.selectbox("Type de rapport",
                ["Bulletin alertes quotidien", "Rapport hebdomadaire", "Synthèse province"])
            include_meteo = st.toggle("Inclure météo par province", value=True)
            include_top   = st.toggle("Inclure top 20 communes", value=True)
        with col_p2:
            st.markdown("**Aperçu du contenu :**")
            st.markdown("- En-tête BeniMeteo + date\n- KPIs alertes ROUGE/ORANGE/JAUNE\n"
                        "- Météo 5 provinces\n- Top 20 communes à risque\n"
                        "- Avertissement légal")

        if st.button("🖨️ Générer le PDF", type="primary"):
            with st.spinner("Génération du rapport PDF..."):
                # Calcul scores communes
                from src.demographics import aggregate_by_commune
                comm_agg = aggregate_by_commune(df_hcp)
                comm_agg["score"] = comm_agg.apply(
                    lambda r: compute_score_v2(r.to_dict(), conditions), axis=1)
                comm_agg["niveau_alerte"] = comm_agg["score"].apply(lambda s: alert_level_v2(s)[0])

                # Météo provinces (cache)
                @st.cache_data(ttl=3600, show_spinner=False)
                def load_prov_data():
                    return fetch_all_provinces(days=3)
                prov_data = load_prov_data() if include_meteo else {}

                snapshot = {
                    "pop_vulnerable_rouge": int(comm_agg[comm_agg["niveau_alerte"]=="ROUGE"]["pop_vulnerable"].sum())
                }

                pdf_bytes = generate_alert_report(
                    comm_agg, prov_data if include_meteo else None,
                    snapshot, title=report_type)

                if pdf_bytes:
                    filename = f"benimeteo_{report_type.replace(' ','_').lower()}_{date.today()}.pdf"
                    st.download_button(
                        "⬇️ Télécharger le rapport PDF",
                        pdf_bytes, filename, "application/pdf",
                        type="primary")
                    save_report_to_file(pdf_bytes, filename)
                    st.success(f"✅ Rapport généré : {filename}")
                else:
                    st.error("Erreur de génération. Vérifiez que reportlab est installé : pip install reportlab")

    with tab_p2:
        rapports = list_reports()
        if rapports:
            for r in rapports:
                col_r1, col_r2, col_r3 = st.columns([3, 1, 1])
                col_r1.markdown(f"📄 **{r['nom']}**")
                col_r2.caption(r["date"])
                col_r3.caption(f"{r['taille']//1024} KB")
        else:
            st.info("Aucun rapport généré. Utilisez l'onglet 'Générer un rapport'.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — ANOMALIES CLIMATIQUES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Anomalies Climatiques":
    st.markdown("## 🔬 Détection d'Anomalies Climatiques")
    st.markdown("Détection automatique des événements météorologiques extrêmes "
                "via Isolation Forest (algorithme non supervisé).")

    from src.models.ml_advanced import detect_anomalies

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        contamination = st.slider("Sensibilité (%)", 1, 15, 5,
                                   help="% d'anomalies attendues dans l'historique")
        variables_sel = st.multiselect("Variables à analyser",
            [c for c in ["tmax","tmin","tmean","temp_mean","precip",
                         "wind_max","humidity_max","pressure"] if c in df_hist.columns],
            default=[c for c in ["tmax","precip","wind_max"] if c in df_hist.columns])
    with col_a2:
        st.markdown("**Interprétation :**")
        st.markdown("- **Score élevé** → événement très inhabituel\n"
                    "- **Canicule** → T°max > 42°C\n"
                    "- **Précipitations extrêmes** → > 50mm/jour\n"
                    "- **Anomalie composite** → combinaison inhabituelle")

    if st.button("🔍 Lancer la détection", type="primary"):
        with st.spinner("Analyse en cours..."):
            anomalies = detect_anomalies(df_hist, variables_sel, contamination/100)

        if len(anomalies) > 0:
            st.success(f"✅ {len(anomalies)} anomalies détectées sur {len(df_hist):,} jours")

            # Graphique temporel
            fig_an = go.Figure()
            # Historique tmax si dispo
            if "tmax" in df_hist.columns:
                fig_an.add_trace(go.Scatter(
                    x=df_hist.index, y=df_hist["tmax"],
                    name="T°Max historique", line=dict(color="#7BB8D4", width=1)))
            # Points anomalies
            if "tmax" in anomalies.columns and "date" in anomalies.columns:
                fig_an.add_trace(go.Scatter(
                    x=anomalies["date"], y=anomalies["tmax"],
                    mode="markers", name="Anomalie détectée",
                    marker=dict(color="#D62828", size=10, symbol="x")))
            fig_an.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#E8D5B7", height=380,
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_an, use_container_width=True)

            # Tableau anomalies
            cols_show = [c for c in ["date","tmax","tmin","precip","wind_max",
                                      "type_anomalie","score_anomalie"] if c in anomalies.columns]
            st.dataframe(anomalies[cols_show].head(50),
                        use_container_width=True, hide_index=True)

            # Téléchargement
            csv_an = anomalies.to_csv(index=False, sep=";", encoding="utf-8-sig")
            st.download_button("⬇️ Exporter les anomalies CSV",
                              csv_an.encode("utf-8-sig"),
                              f"anomalies_{date.today()}.csv", "text/csv")
        else:
            st.info("Aucune anomalie détectée avec ces paramètres.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — CALIBRATION SCIENTIFIQUE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Calibration Scientifique":
    st.markdown("## 🔬 Calibration Scientifique du Score V2")
    st.markdown("""
    <div class="alert-orange">
    Trois méthodes scientifiques pour valider et améliorer les coefficients du score :
    <b>Régression Logistique</b> sur événements historiques ·
    <b>Méthode Delphi</b> (consensus 5 experts) ·
    <b>HUMEX</b> (Human Exposure Index — OMS)
    </div>
    """, unsafe_allow_html=True)

    from src.calibration import (calculer_consensus_delphi, calculer_humex,
                                   compute_score_v3, generer_rapport_calibration,
                                   comparer_score_vs_humex, calibrer_par_regression,
                                   EVENEMENTS_HISTORIQUES, EXPERTS_DELPHI)

    tab_c1, tab_c2, tab_c3, tab_c4 = st.tabs([
        "📊 Régression Logistique",
        "👥 Méthode Delphi",
        "🌡 Comparaison HUMEX",
        "⚖️ V2 vs V3"
    ])

    # ── TAB 1 : Régression Logistique ─────────────────────────────────────────
    with tab_c1:
        st.markdown("### Calibration par Régression Logistique Ordinale")
        st.markdown("""
        **Principe** : Pour chaque événement climatique historique documenté dans la région,
        on connaît le niveau d'alerte réel déclenché (ROUGE/ORANGE/JAUNE/VERT).
        La régression logistique trouve les coefficients qui maximisent la correspondance
        entre les conditions météo observées et le niveau réel.

        **Source des événements** : Rapports Protection Civile Maroc + Bulletins DMN 2020-2023.
        """)

        st.markdown("#### Événements historiques utilisés")
        ev_rows = []
        for label, s, e, niveau, tmax, precip, desc in EVENEMENTS_HISTORIQUES:
            color = {"ROUGE":"🔴","ORANGE":"🟠","JAUNE":"🟡","VERT":"🟢"}[niveau]
            ev_rows.append({
                "Événement": label,
                "Période": f"{s} → {e}",
                "Niveau réel": f"{color} {niveau}",
                "Tmax (°C)": tmax,
                "Précip (mm)": precip,
                "Description": desc,
            })
        st.dataframe(pd.DataFrame(ev_rows), use_container_width=True, hide_index=True)

        if st.button("▶️ Lancer la régression logistique", type="primary"):
            with st.spinner("Entraînement du modèle..."):
                reg = calibrer_par_regression(df_hist, aggregate_by_commune(df_hcp))

            st.success(f"✅ Précision globale : **{reg['accuracy']}%** "
                       f"sur {reg['nb_evenements']} événements")

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.markdown("**F1-Score par niveau d'alerte :**")
                for niveau, f1 in reg['rapport_classes'].items():
                    emoji = {"ROUGE":"🔴","ORANGE":"🟠","JAUNE":"🟡","VERT":"🟢"}[niveau]
                    st.metric(f"{emoji} {niveau}", f"{f1}%")

            with col_r2:
                st.markdown("**Prédictions vs réalité :**")
                pred_rows = []
                for p in reg['predictions']:
                    ok = "✅" if p['correct'] else "❌"
                    pred_rows.append({
                        "Événement": p['label'][:25],
                        "Réel": p['reel'],
                        "Prédit": p['predit'],
                        "Résultat": ok
                    })
                st.dataframe(pd.DataFrame(pred_rows),
                             use_container_width=True, hide_index=True)

            st.markdown("**Poids des facteurs (régression) :**")
            poids_df = pd.DataFrame([
                {"Facteur": "Chaleur (contribution)", "Poids": f"{reg['poids_chaleur']*100:.1f}%"},
                {"Facteur": "Crues (contribution)",   "Poids": f"{reg['poids_crues']*100:.1f}%"},
            ])
            st.dataframe(poids_df, use_container_width=True, hide_index=True)

    # ── TAB 2 : Méthode Delphi ────────────────────────────────────────────────
    with tab_c2:
        st.markdown("### Méthode Delphi — Consensus d'Experts")
        st.markdown("""
        **Principe** : 5 experts de la région ont estimé l'importance relative de chaque
        facteur de risque. Le consensus (moyenne + intervalle de confiance 95%) donne des
        coefficients validés par des professionnels de terrain.

        **Source** : Profils d'experts région Béni Mellal-Khénifra
        (Protection Civile, CHR, DMN, Santé Publique, ANDZOA).
        """)

        delphi = calculer_consensus_delphi()

        st.markdown("#### Experts consultés")
        expert_rows = []
        for expert, data in EXPERTS_DELPHI.items():
            expert_rows.append({
                "Expert": expert,
                "Domaine": data["domaine"],
                "Poids chaleur": f"{data['poids_chaleur']*100:.0f}%",
                "Poids crues": f"{data['poids_crues']*100:.0f}%",
                "Poids froid": f"{data['poids_froid']*100:.0f}%",
                "Commentaire": data["commentaire"][:60] + "...",
            })
        st.dataframe(pd.DataFrame(expert_rows),
                     use_container_width=True, hide_index=True)

        st.markdown("#### Consensus Delphi — Facteurs météo")
        facteurs = ['poids_chaleur','poids_froid','poids_crues',
                    'poids_vent','poids_uv','poids_secheresse']
        labels   = ['Chaleur','Froid','Crues','Vent','UV','Sécheresse']
        cons = delphi['consensus']

        cons_rows = []
        for f, label in zip(facteurs, labels):
            c = cons[f]
            cons_rows.append({
                "Facteur": label,
                "Moyenne": f"{c['moyenne']*100:.1f}%",
                "IC 95% bas": f"{c['ic_low']*100:.1f}%",
                "IC 95% haut": f"{c['ic_high']*100:.1f}%",
                "Accord": c['accord'],
                "CV": f"{c['cv_pct']:.1f}%",
            })
        st.dataframe(pd.DataFrame(cons_rows),
                     use_container_width=True, hide_index=True)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("#### Coefficients démographiques (Delphi)")
            d_coef = delphi['coefficients_demo_delphi']
            st.metric("Poids âgés 60+", f"{d_coef['coeff_60plus']*100:.1f}%",
                      "± IC 95%")
            st.metric("Poids enfants < 15", f"{d_coef['coeff_moins15']*100:.1f}%")
            st.metric("Poids chroniques", f"{d_coef['coeff_chroniques']*100:.1f}%")
        with col_d2:
            st.markdown("#### Facteur isolement médical (Delphi)")
            iso_c = cons['coeff_isolement_60km']
            st.metric("Coeff. > 60 km", f"× {iso_c['moyenne']:.2f}",
                      f"IC [{iso_c['ic_low']:.2f} ; {iso_c['ic_high']:.2f}]")
            st.caption("Accord entre experts : " + iso_c['accord'])

        st.markdown("#### Commentaires des experts")
        for expert, comment in delphi['commentaires_experts'].items():
            st.info(f"**{expert}** : {comment}")

    # ── TAB 3 : HUMEX ─────────────────────────────────────────────────────────
    with tab_c3:
        st.markdown("### Comparaison avec HUMEX (Human Exposure Index)")
        st.markdown("""
        **HUMEX** est l'indice d'exposition humaine de l'OMS/GIEC.
        Il mesure la vulnérabilité réelle selon 3 composantes :

        `HUMEX = Exposition × Sensibilité × (1 / Capacité_adaptation)`

        **Source** : Kovats & Hajat (2008). *Epidemiologic Reviews* +
        Birkmann et al. (2013). *Natural Hazards*.
        """)

        temp_humex = st.slider("Température de test (°C)", 25, 50, 42)
        cond_humex = {'temp': temp_humex, 'precip': 0, 'wind': 15, 'humidity': 35, 'uv': 9}

        # Calculer pour commune type
        comm_type = {'population': 5000, 'pct_60_plus': 12.0,
                     'pct_moins_15': 34.0, 'altitude': 480}
        h = calculer_humex(5000, 12.0, 34.0, float(temp_humex), 480)

        col_h1, col_h2, col_h3, col_h4 = st.columns(4)
        col_h1.metric("HUMEX Score", f"{h['humex_score']}/100")
        col_h2.metric("Exposition", f"{h['exposition']:.3f}")
        col_h3.metric("Sensibilité", f"{h['sensibilite']:.3f}")
        col_h4.metric("Capacité adapt.", f"{h['capacite_adapt']:.3f}")
        st.caption(f"Interprétation : {h['interpretation']}")
        st.caption(f"Source : {h['source']}")

        st.markdown("#### Corrélation Score V2 vs HUMEX — 119 communes")
        with st.spinner("Calcul comparaison V2 vs HUMEX pour 119 communes..."):
            from src.alert_engine import compute_score_v2, get_seasonal_context as gsc
            comm_agg = aggregate_by_commune(df_hcp)
            def sf(e, c): return compute_score_v2(e, c, seasonal_context=gsc())
            df_humex_comp = comparer_score_vs_humex(comm_agg, cond_humex, sf)

        corr_val = df_humex_comp['score_v2'].corr(df_humex_comp['humex'])
        st.metric("Corrélation Pearson Score V2 ↔ HUMEX",
                  f"r = {corr_val:.3f}",
                  "Corrélation forte si r > 0.7")

        st.dataframe(
            df_humex_comp[['commune','province','score_v2','humex','ecart']].head(20),
            use_container_width=True, hide_index=True)

        st.caption("Écart faible = Score V2 cohérent avec HUMEX (OMS)")

    # ── TAB 4 : V2 vs V3 ──────────────────────────────────────────────────────
    with tab_c4:
        st.markdown("### Comparaison Score V2 (actuel) vs Score V3 (Delphi)")
        st.markdown("""
        Le **Score V3** utilise les coefficients démographiques issus du consensus Delphi
        au lieu des valeurs empiriques initiales.
        """)

        delphi = calculer_consensus_delphi()
        from src.alert_engine import compute_score_v2, alert_level_v2, get_seasonal_context as gsc
        seasonal = gsc()
        sample = aggregate_by_commune(df_hcp).iloc[0].to_dict()

        comp_rows = []
        for label, s, e, niveau_reel, tmax, precip, desc in EVENEMENTS_HISTORIQUES:
            cond  = {'temp':tmax,'precip':precip,'wind':15,'humidity':40,'uv':7}
            sc_v2 = compute_score_v2(sample, cond, seasonal_context=seasonal)
            sc_v3 = compute_score_v3(sample, cond, seasonal_context=seasonal,
                                      delphi_consensus=delphi)
            lvl_v2 = alert_level_v2(sc_v2)[0]
            lvl_v3 = alert_level_v2(sc_v3)[0]
            em = {"ROUGE":"🔴","ORANGE":"🟠","JAUNE":"🟡","VERT":"🟢"}
            comp_rows.append({
                "Événement":   label,
                "Niveau réel": f"{em[niveau_reel]} {niveau_reel}",
                "Score V2":    sc_v2,
                "Niveau V2":   f"{em[lvl_v2]} {lvl_v2}",
                "Score V3":    sc_v3,
                "Niveau V3":   f"{em[lvl_v3]} {lvl_v3}",
                "V2 ✓":        "✅" if lvl_v2 == niveau_reel else "❌",
                "V3 ✓":        "✅" if lvl_v3 == niveau_reel else "❌",
            })

        df_comp = pd.DataFrame(comp_rows)
        acc_v2 = df_comp["V2 ✓"].eq("✅").mean() * 100
        acc_v3 = df_comp["V3 ✓"].eq("✅").mean() * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Précision Score V2", f"{acc_v2:.1f}%")
        c2.metric("Précision Score V3 (Delphi)", f"{acc_v3:.1f}%")
        c3.metric("Amélioration", f"{acc_v3-acc_v2:+.1f}%")

        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        st.markdown("#### Note méthodologique")
        st.info(
            "Les écarts entre V2 et V3 s'expliquent principalement par l'utilisation "
            "de données simulées (DMN) plutôt que de vraies données de stations. "
            "Avec **pip install meteostat** et le cache des vraies données, la précision "
            "augmentera significativement car les valeurs extrêmes documentées (48-49°C en 2021) "
            "seront correctement reproduites dans l'historique d'entraînement."
        )



# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — 💧 PLANIFICATION RÉSERVOIRS EAU
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💧 Besoins en Eau":
    # ════════════════════════════════════════════════════════════════════════
    # PAGE PLANIFICATION RÉSERVOIRS D'EAU
    # Paramètres : 5L/pers/j potable + 35L/pers/j autres = 40L/pers/j total
    # Sources : OMS (2011) · Sphere Handbook (2018) · ONEE Maroc · RGPH 2024
    # ════════════════════════════════════════════════════════════════════════
    from src.water_planning import (calculer_region, stats_region,
                                     generer_csv, generer_pdf,
                                     EAU_POTABLE_L_PERS_J, EAU_AUTRES_L_PERS_J,
                                     EAU_TOTAL_L_PERS_J, SEUIL_STRESS_THERMIQUE,
                                     FACTEUR_STRESS_THERMIQUE, SEUIL_ISOLEMENT_KM,
                                     VOLUME_RESERVOIR_M3)
    import plotly.express as px
    import plotly.graph_objects as go

    st.markdown("## 💧 Planification des Réservoirs d'Eau")
    st.markdown("""
    <div class="alert-vert">
    <b>Base de calcul :</b>
    5 L/pers/jour (eau potable OMS) + 35 L/pers/jour (usage domestique) = <b>40 L/pers/jour</b> total.
    Coefficients : stress thermique Tmax > 40°C → ×1.25 · isolement > 60km → autonomie 48h · source naturelle → −50% hors-boisson.
    </div>
    """, unsafe_allow_html=True)

    # ── Panneau de contrôle ─────────────────────────────────────────────────
    st.markdown("### Paramètres de simulation")
    col_p1, col_p2, col_p3, col_p4 = st.columns([2, 1, 1, 1])

    with col_p1:
        tmax_sim = st.slider(
            "🌡 Température max prévue (°C)",
            min_value=10, max_value=55, value=32,
            help=f"Au-delà de {SEUIL_STRESS_THERMIQUE}°C : besoins ×{FACTEUR_STRESS_THERMIQUE} (stress thermique OMS 2008)"
        )
        if tmax_sim > SEUIL_STRESS_THERMIQUE:
            st.markdown(
                f'<div class="alert-orange">⚠️ Tmax={tmax_sim}°C > {SEUIL_STRESS_THERMIQUE}°C — '
                f'Facteur stress thermique ×{FACTEUR_STRESS_THERMIQUE} activé</div>',
                unsafe_allow_html=True
            )

    with col_p2:
        source_nat = st.checkbox(
            "Source naturelle (riviere / source)",
            value=False,
            help="Réduit de 50% les besoins hors-boisson si une source est présente"
        )
        if source_nat:
            st.caption("−50% hors-boisson activé")

    with col_p3:
        budget_dh = st.number_input(
            "💰 Budget (DH)",
            min_value=0, max_value=200_000_000,
            value=10_000_000, step=1_000_000,
            format="%d"
        )

    with col_p4:
        nb_afficher = st.selectbox(
            "Afficher top",
            [50, 100, 200, 500, 2485],
            index=0
        )

    # ── Calcul ──────────────────────────────────────────────────────────────
    with st.spinner("Calcul des besoins pour 2 485 douars..."):
        df_w = calculer_region(df_hcp, tmax=tmax_sim,
                                source_naturelle_global=source_nat)
        st_r  = stats_region(df_w)

    # ── Métriques principales ────────────────────────────────────────────────
    st.markdown("### Résumé région Béni Mellal-Khénifra")
    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    mc1.metric("💧 Eau potable/j",
               f"{st_r.get('eau_potable_m3_j',0):.0f} m³",
               f"{EAU_POTABLE_L_PERS_J} L/pers")
    mc2.metric("🚿 Eau usage/j",
               f"{st_r.get('eau_autres_m3_j',0):.0f} m³",
               f"{EAU_AUTRES_L_PERS_J} L/pers")
    mc3.metric("🏺 Total eau/j",
               f"{st_r.get('eau_totale_m3_j',0):.0f} m³",
               f"{EAU_TOTAL_L_PERS_J} L/pers × 1.25" if tmax_sim > SEUIL_STRESS_THERMIQUE else f"{EAU_TOTAL_L_PERS_J} L/pers")
    mc4.metric("🏗 Réservoirs 5m³",
               f"{st_r.get('nb_reservoirs_total',0):,}")
    mc5.metric("💰 Coût estimé",
               f"{st_r.get('cout_total_dh',0)/1e6:.1f}M DH")
    mc6.metric("🗺 Isolés >60km",
               f"{st_r.get('nb_isoles',0):,}",
               "autonomie 48h")

    # Niveaux de priorité
    st.markdown("### Niveaux de priorité d'intervention")
    col_niv = st.columns(4)
    niveaux = [
        ("🔴 CRITIQUE", st_r.get("nb_critique",0), "#FEE2E2", "#DC2626"),
        ("🟠 HAUTE",    st_r.get("nb_haute",0),    "#FEF3C7", "#EA580C"),
        ("🟡 MOYENNE",  st_r.get("nb_moyenne",0),  "#FFF9C4", "#CA8A04"),
        ("🟢 NORMALE",  st_r.get("nb_normale",0),  "#DCFCE7", "#16A34A"),
    ]
    for col, (lab, n, bg, fg) in zip(col_niv, niveaux):
        pop_niv = df_w[df_w["niveau_priorite"] == lab.split()[-1]]["population"].sum() if len(df_w) > 0 else 0
        col.markdown(f"""
        <div style="background:{bg};border-radius:10px;padding:16px;text-align:center;
                    border-left:4px solid {fg}">
            <div style="font-size:13px;font-weight:600;color:{fg}">{lab}</div>
            <div style="font-size:28px;font-weight:700;color:{fg};margin:4px 0">{n}</div>
            <div style="font-size:11px;color:{fg};opacity:0.8">douars</div>
            <div style="font-size:11px;color:{fg};margin-top:4px">{pop_niv:,} hab.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── 3 ONGLETS PRINCIPAUX ─────────────────────────────────────────────────
    tab_map, tab_table, tab_budget = st.tabs([
        "🗺️ Carte des douars prioritaires",
        "📋 Tableau interactif",
        "💰 Plan budgété",
    ])

    # ── ONGLET 1 : CARTE ─────────────────────────────────────────────────────
    with tab_map:
        st.markdown("#### Carte — Douars prioritaires (rouge = CRITIQUE, orange = HAUTE)")
        st.caption("Taille des cercles proportionnelle au nombre de réservoirs nécessaires")

        df_map = df_w[df_w["lat"].notna() & df_w["lon"].notna()].copy()

        if len(df_map) > 0:
            # Filtre niveau pour carte
            niveaux_map = st.multiselect(
                "Niveaux à afficher sur la carte",
                ["CRITIQUE", "HAUTE", "MOYENNE", "NORMALE"],
                default=["CRITIQUE", "HAUTE"],
                key="map_niveaux"
            )
            if niveaux_map:
                df_map = df_map[df_map["niveau_priorite"].isin(niveaux_map)]

            fig_map = px.scatter_mapbox(
                df_map,
                lat="lat", lon="lon",
                color="niveau_priorite",
                size="nb_reservoirs_5m3",
                size_max=20,
                color_discrete_map={
                    "CRITIQUE": "#DC2626",
                    "HAUTE":    "#EA580C",
                    "MOYENNE":  "#CA8A04",
                    "NORMALE":  "#16A34A",
                },
                hover_name="douar",
                hover_data={
                    "province": True,
                    "commune": True,
                    "population": True,
                    "besoin_total_L_j": True,
                    "nb_reservoirs_5m3": True,
                    "cout_estime_dh": True,
                    "dist_centre_sante_km": True,
                    "niveau_priorite": True,
                    "lat": False, "lon": False,
                },
                labels={
                    "niveau_priorite": "Priorité",
                    "nb_reservoirs_5m3": "Nb réservoirs",
                    "besoin_total_L_j": "Besoin total L/j",
                    "cout_estime_dh": "Coût DH",
                    "dist_centre_sante_km": "Dist. CHR (km)",
                },
                zoom=7,
                center={"lat": 32.337, "lon": -6.349},
                mapbox_style="carto-positron",
                height=550,
            )
            fig_map.update_layout(
                margin={"r":0,"t":0,"l":0,"b":0},
                legend_title_text="Priorité",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_map, use_container_width=True)
            st.caption(f"{len(df_map):,} douars affichés · "
                       f"Données RGPH 2024 HCP + OpenStreetMap + calcul BeniMeteo")
        else:
            # Fallback st.map si pas de données
            st.map(df_w[["lat","lon"]].dropna().rename(
                columns={"lat":"latitude","lon":"longitude"}))

    # ── ONGLET 2 : TABLEAU ───────────────────────────────────────────────────
    with tab_table:
        st.markdown("#### Tableau interactif — Tous les douars")

        # Filtres
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            prov_sel = st.multiselect(
                "Province",
                sorted(df_w["province"].dropna().unique()),
                default=[]
            )
        with col_f2:
            # Communes disponibles selon province
            df_comm = df_w if not prov_sel else df_w[df_w["province"].isin(prov_sel)]
            comm_sel = st.multiselect(
                "Commune",
                sorted(df_comm["commune"].dropna().unique()),
                default=[]
            )
        with col_f3:
            niv_sel = st.multiselect(
                "Niveau priorité",
                ["CRITIQUE", "HAUTE", "MOYENNE", "NORMALE"],
                default=[]
            )
        with col_f4:
            pop_min = st.number_input("Pop. minimale", 0, 50000, 0, 100)

        # Appliquer filtres
        df_filt = df_w.copy()
        if prov_sel:  df_filt = df_filt[df_filt["province"].isin(prov_sel)]
        if comm_sel:  df_filt = df_filt[df_filt["commune"].isin(comm_sel)]
        if niv_sel:   df_filt = df_filt[df_filt["niveau_priorite"].isin(niv_sel)]
        if pop_min:   df_filt = df_filt[df_filt["population"] >= pop_min]

        df_display = df_filt.head(nb_afficher)

        st.caption(f"Affichage : {len(df_display):,} douars sur {len(df_filt):,} filtrés ({len(df_w):,} total)")

        # Colonnes à afficher
        cols_show = [
            "rang", "niveau_priorite", "province", "commune", "douar",
            "population", "nb_menages", "pct_moins_15", "pct_60_plus", "pct_vulnerable",
            "dist_centre_sante_km", "isole", "altitude_m",
            "besoin_potable_L_j", "besoin_autres_L_j", "besoin_total_L_j",
            "L_par_pers_j", "jours_autonomie",
            "volume_total_m3", "nb_reservoirs_5m3", "cout_estime_dh", "score_priorite"
        ]
        cols_show = [c for c in cols_show if c in df_display.columns]

        st.dataframe(
            df_display[cols_show].rename(columns={
                "rang": "Rang", "niveau_priorite": "Priorité",
                "province": "Province", "commune": "Commune", "douar": "Douar",
                "population": "Pop.", "nb_menages": "Ménages",
                "pct_moins_15": "% -15ans", "pct_60_plus": "% 60+",
                "pct_vulnerable": "% Vuln.",
                "dist_centre_sante_km": "Dist. CHR (km)", "isole": "Isolé",
                "altitude_m": "Alt. (m)",
                "besoin_potable_L_j": "Potable L/j",
                "besoin_autres_L_j": "Autres L/j",
                "besoin_total_L_j": "Total L/j",
                "L_par_pers_j": "L/pers/j",
                "jours_autonomie": "Autonomie (j)",
                "volume_total_m3": "Volume m³",
                "nb_reservoirs_5m3": "Rés. 5m³",
                "cout_estime_dh": "Coût DH",
                "score_priorite": "Score",
            }),
            use_container_width=True,
            hide_index=True,
            height=450,
        )

        # ── Exports ─────────────────────────────────────────────────────────
        st.markdown("#### Exporter le plan d'intervention")
        col_exp1, col_exp2, col_exp3 = st.columns(3)

        with col_exp1:
            csv_bytes = generer_csv(df_filt)
            st.download_button(
                label="⬇️ CSV — Liste terrain complète",
                data=csv_bytes,
                file_name=f"plan_eau_BM_Khenifra_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                help="Fichier CSV séparé par ; — compatible Excel et LibreOffice"
            )

        with col_exp2:
            with st.spinner("Génération PDF..."):
                pdf_bytes = generer_pdf(df_filt, st_r, tmax_sim, source_nat)
            if pdf_bytes:
                st.download_button(
                    label="📄 PDF — Bulletin terrain (Top 50)",
                    data=pdf_bytes,
                    file_name=f"bulletin_eau_BM_Khenifra_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    help="PDF A4 paysage — tableau coloré par niveau de priorité"
                )
            else:
                st.warning("ReportLab non installé — pip install reportlab")

        with col_exp3:
            # CSV filtré seulement douars CRITIQUE et HAUTE
            df_urgents = df_filt[df_filt["niveau_priorite"].isin(["CRITIQUE","HAUTE"])]
            csv_urgents = generer_csv(df_urgents)
            st.download_button(
                label="🚨 CSV — Urgences seulement",
                data=csv_urgents,
                file_name=f"urgences_eau_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                help=f"Seulement les douars CRITIQUE et HAUTE ({len(df_urgents)} douars)"
            )

    # ── ONGLET 3 : PLAN BUDGÉTÉ ──────────────────────────────────────────────
    with tab_budget:
        st.markdown(f"#### Plan d'intervention — Budget {budget_dh:,} DH")
        st.caption("Coûts estimatifs : réservoir 5m³ = 8 000 DH + installation 2 000 DH + transport 50 DH/km")

        df_budget = df_w.copy()
        df_budget["cout_cumule_dh"] = df_budget["cout_estime_dh"].cumsum()
        df_budget["dans_budget"]    = df_budget["cout_cumule_dh"] <= budget_dh

        n_couverts  = int(df_budget["dans_budget"].sum())
        pop_couverte= int(df_budget[df_budget["dans_budget"]]["population"].sum())
        cout_total  = int(df_budget[df_budget["dans_budget"]]["cout_estime_dh"].sum())
        reste       = budget_dh - cout_total

        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        col_b1.metric("Douars couverts", f"{n_couverts:,}", f"sur {len(df_w):,}")
        col_b2.metric("Population couverte", f"{pop_couverte:,} hab.")
        col_b3.metric("Coût réel", f"{cout_total:,} DH")
        col_b4.metric("Budget restant", f"{reste:,} DH",
                       delta_color="normal" if reste >= 0 else "inverse")

        # Barre de progression budget
        pct_utilise = min(cout_total / max(budget_dh, 1) * 100, 100)
        st.progress(int(pct_utilise),
                    text=f"Budget utilisé : {pct_utilise:.1f}%")

        # Tableau plan budgété
        df_plan = df_budget[df_budget["dans_budget"]][[
            "rang", "niveau_priorite", "province", "commune", "douar",
            "population", "dist_centre_sante_km", "isole",
            "besoin_total_L_j", "nb_reservoirs_5m3",
            "cout_estime_dh", "cout_cumule_dh", "score_priorite"
        ]].rename(columns={
            "rang": "Rang", "niveau_priorite": "Priorité",
            "province": "Province", "commune": "Commune", "douar": "Douar",
            "population": "Pop.", "dist_centre_sante_km": "Dist. CHR",
            "isole": "Isolé", "besoin_total_L_j": "Besoin L/j",
            "nb_reservoirs_5m3": "Rés. 5m³",
            "cout_estime_dh": "Coût DH", "cout_cumule_dh": "Cumulé DH",
            "score_priorite": "Score"
        })
        st.dataframe(df_plan, use_container_width=True, hide_index=True, height=400)

        # Export plan budgété
        csv_budget = generer_csv(df_budget[df_budget["dans_budget"]])
        st.download_button(
            "⬇️ Exporter plan budgété (CSV)",
            data=csv_budget,
            file_name=f"plan_budget_{budget_dh//1000}k_DH_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

        with st.spinner("Génération PDF plan budgété..."):
            pdf_budget = generer_pdf(df_budget[df_budget["dans_budget"]], st_r, tmax_sim, source_nat)
        if pdf_budget:
            st.download_button(
                "📄 Bulletin PDF plan budgété",
                data=pdf_budget,
                file_name=f"bulletin_budget_{budget_dh//1000}k_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )

    # ── Note méthodologique ─────────────────────────────────────────────────
    with st.expander("📖 Méthodologie et sources scientifiques"):
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("""
**Besoins en eau (paramètres fixes)**
- Eau potable : **5 L/pers/jour** (OMS 2011 — minimum urgence humanitaire)
- Usage général : **35 L/pers/jour** (hygiène + cuisine + animaux + agriculture)
- **Total baseline : 40 L/pers/jour**

**Coefficients de pondération**
- **Stress thermique** (Tmax > 40°C) : besoins × 1.25 — déshydratation accrue
- **Isolement** (dist. > 60km) : autonomie 48h au lieu de 24h — stockage doublé
- **Source naturelle** : −50% sur les besoins hors-boisson

**Priorité d'intervention (score 0-100)**
- Vulnérabilité démographique : 40% (% personnes âgées + enfants)
- Isolement médical : 30% (distance centre de santé)
- Stress thermique : 20% (Tmax > 40°C)
- Altitude : 10% (accès difficile)
            """)
        with col_m2:
            st.markdown("""
**Réservoirs**
- Volume unitaire de référence : **5 m³**
- Autonomie standard : 24h (1 jour)
- Autonomie isolement : 48h (2 jours) si dist > 60km

**Coûts estimatifs (ONEE + BTP Maroc 2024)**
- Réservoir plastique 5m³ : **8 000 DH**
- Installation : **2 000 DH**
- Transport : **50 DH/km** depuis CHR

**Sources scientifiques**
- OMS (2011) *Domestic Water Quantity, Service Level and Health*
- Sphere Handbook (2018) — Chapitre WASH
- OMS (2008) *Heat Health Action Plans*
- ONEE Maroc — Niveaux de service eau potable rural
- FAO (2006) — Besoins eau animaux et irrigation
- RGPH 2024 HCP Maroc — 2 485 douars région BM-Khénifra
            """)


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;padding:20px;margin-top:40px;border-top:1px solid #E0E4EA;
            font-size:12px;color:#9CA3AF;">
    <b>BeniMeteo</b> · Université Sultan Moulay Slimane · ENSA Béni Mellal ·
    Région Béni Mellal-Khénifra, Maroc<br>
    Météo : <b>Open-Meteo</b> (ERA5) · <b>Meteostat</b> (stations réelles) ·
    Démographie : <b>HCP RGPH 2024</b> · <b>OpenStreetMap</b> (hôpitaux) |
    Modèles : SARIMA · XGBoost · Prophet · LSTM · Isolation Forest |
    Alertes : SQLite + CSV · Email SMTP · Validation terrain
</div>
""", unsafe_allow_html=True)
