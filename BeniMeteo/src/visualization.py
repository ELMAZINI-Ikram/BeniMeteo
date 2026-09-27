"""
visualization.py — Graphiques Plotly réutilisables.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

COLORS = {
    "primary":   "#7BB8D4",
    "secondary": "#C4622D",
    "green":     "#4A7C59",
    "yellow":    "#FCBF49",
    "danger":    "#D62828",
    "bg":        "rgba(0,0,0,0)",
    "grid":      "rgba(255,255,255,0.06)",
    "text":      "#E8D5B7",
}

BASE_LAYOUT = dict(
    plot_bgcolor=COLORS["bg"], paper_bgcolor=COLORS["bg"],
    font_color=COLORS["text"],
    margin=dict(l=50, r=20, t=40, b=50),
    xaxis=dict(gridcolor=COLORS["grid"], showline=False),
    yaxis=dict(gridcolor=COLORS["grid"], showline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", font_color=COLORS["text"]),
)


def temperature_series(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "tmax" in df.columns and "tmin" in df.columns:
        fig.add_trace(go.Scatter(
            x=list(df.index) + list(df.index[::-1]),
            y=list(df["tmax"]) + list(df["tmin"][::-1]),
            fill="toself", fillcolor="rgba(123,184,212,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="Plage T°",
        ))
    if "temp_mean" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["temp_mean"], mode="lines",
                                  name="T° Moyenne", line=dict(color=COLORS["primary"], width=2)))
    if "tmax" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["tmax"], mode="lines",
                                  name="T° Max", line=dict(color=COLORS["secondary"], width=1.2, dash="dot")))
    if "tmin" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["tmin"], mode="lines",
                                  name="T° Min", line=dict(color="#7BB8D4", width=1.2, dash="dot")))
    fig.update_layout(**BASE_LAYOUT, title="Évolution des Températures", height=360)
    return fig


def precipitation_bar(df: pd.DataFrame) -> go.Figure:
    monthly = df.resample("ME")["precip"].sum().reset_index()
    monthly.columns = ["date", "precip"]
    fig = go.Figure(go.Bar(x=monthly["date"], y=monthly["precip"],
                            marker_color=COLORS["primary"], marker_line_width=0))
    fig.update_layout(**BASE_LAYOUT, title="Précipitations Mensuelles (mm)", height=300)
    return fig


def wind_rose(df: pd.DataFrame) -> go.Figure:
    dirs = {0: "N", 45: "NE", 90: "E", 135: "SE", 180: "S", 225: "SW", 270: "W", 315: "NW"}
    df_w = df.copy()
    df_w["dir_label"] = df_w["wind_dir"].map(
        lambda x: dirs.get(int(round(float(x) / 45) * 45) % 360, "N") if pd.notna(x) else "N")
    counts = df_w.groupby("dir_label").size().reset_index(name="count")
    fig = px.bar_polar(counts, r="count", theta="dir_label", color="count",
                       color_continuous_scale=[COLORS["bg"], COLORS["primary"], COLORS["secondary"]])
    fig.update_layout(plot_bgcolor=COLORS["bg"], paper_bgcolor=COLORS["bg"],
                      font_color=COLORS["text"], height=300,
                      margin=dict(l=10, r=10, t=30, b=10),
                      polar=dict(bgcolor="rgba(0,0,0,0)"),
                      coloraxis_showscale=False)
    return fig


def heatmap_monthly(df: pd.DataFrame, variable: str = "temp_mean", title: str = "") -> go.Figure:
    MONTHS = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun", "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]
    pivot = df.groupby(["year", "month"])[variable].mean().unstack()
    pivot.columns = [MONTHS[c - 1] for c in pivot.columns]
    fig = px.imshow(pivot, color_continuous_scale="RdYlBu_r",
                    labels=dict(x="Mois", y="Année", color=variable),
                    title=title or f"Heatmap — {variable}")
    fig.update_layout(plot_bgcolor=COLORS["bg"], paper_bgcolor=COLORS["bg"],
                      font_color=COLORS["text"], height=320,
                      margin=dict(l=60, r=20, t=50, b=40))
    return fig


def forecast_chart(hist: pd.Series, pred_mean: pd.Series,
                   pred_lower=None, pred_upper=None,
                   model_name: str = "Prédiction") -> go.Figure:
    fig = go.Figure()
    hist_tail = hist.iloc[-365:]
    fig.add_trace(go.Scatter(x=hist_tail.index, y=hist_tail, mode="lines",
                              name="Historique", line=dict(color=COLORS["primary"], width=2)))
    if pred_lower is not None and pred_upper is not None:
        fig.add_trace(go.Scatter(
            x=list(pred_mean.index) + list(pred_mean.index[::-1]),
            y=list(pred_upper) + list(pred_lower[::-1]),
            fill="toself", fillcolor="rgba(196,98,45,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="IC 95%"))
    fig.add_trace(go.Scatter(x=pred_mean.index, y=pred_mean, mode="lines+markers",
                              name=model_name,
                              line=dict(color=COLORS["secondary"], width=2.5, dash="dash"),
                              marker=dict(size=5, symbol="diamond")))
    # add_vline est buggé sur certaines versions Plotly — on utilise add_shape à la place
    split_str = str(hist.index[-1])
    fig.add_shape(type="line", x0=split_str, x1=split_str, y0=0, y1=1,
                  xref="x", yref="paper",
                  line=dict(color="rgba(255,255,255,0.3)", dash="dot", width=1.5))
    fig.add_annotation(x=split_str, y=1, xref="x", yref="paper",
                       text="Aujourd'hui", showarrow=False,
                       font=dict(color="rgba(255,255,255,0.5)", size=11),
                       xanchor="left", yanchor="top")
    fig.update_layout(**BASE_LAYOUT, title=f"Prévision — {model_name}", height=380)
    return fig


def multi_variable_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=["Température (°C)", "Précipitations (mm)", "Humidité (%)", "Vent (km/h)"],
                        vertical_spacing=0.06)
    palette = [COLORS["primary"], COLORS["secondary"], "#7BB8D4"]
    for i, col in enumerate([c for c in ["temp_mean", "tmax", "tmin"] if c in df.columns]):
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col,
                                  line=dict(color=palette[i], width=1.5)), row=1, col=1)
    if "precip" in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df["precip"],
                              marker_color=COLORS["primary"], name="Précip.", opacity=0.7), row=2, col=1)
    if "humidity_mean" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["humidity_mean"],
                                  line=dict(color=COLORS["green"], width=1.5), name="Humidité"), row=3, col=1)
    if "wind_max" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["wind_max"],
                                  line=dict(color=COLORS["yellow"], width=1.5), name="Vent max"), row=4, col=1)
    fig.update_layout(plot_bgcolor=COLORS["bg"], paper_bgcolor=COLORS["bg"],
                      font_color=COLORS["text"], height=700,
                      margin=dict(l=50, r=20, t=40, b=40),
                      showlegend=True, legend=dict(bgcolor="rgba(0,0,0,0)"))
    for i in range(1, 5):
        fig.update_xaxes(gridcolor=COLORS["grid"], row=i, col=1)
        fig.update_yaxes(gridcolor=COLORS["grid"], row=i, col=1)
    return fig


def vulnerability_map_plotly(df_zones: pd.DataFrame, conditions: dict) -> go.Figure:
    """Carte Plotly des zones vulnérables (fallback si folium indisponible)."""
    from src.alerts import compute_vulnerability_score, alert_level

    color_map = {"ROUGE": "#D62828", "ORANGE": "#F77F00", "JAUNE": "#FCBF49", "VERT": "#4A7C59"}
    fig = go.Figure()

    for _, z in df_zones.iterrows():
        sc, vp, _ = compute_vulnerability_score(z.to_dict(), conditions)
        lvl, em, _ = alert_level(sc)
        col  = color_map.get(lvl, "#7BB8D4")
        size = max(14, sc / 3.5)
        fig.add_trace(go.Scattermapbox(
            lat=[z["lat"]], lon=[z["lon"]], mode="markers",
            marker=dict(size=size, color=col, opacity=0.85),
            name=z["nom"],
            hovertemplate=(
                f"<b>{z['nom']}</b><br>"
                f"Score: {sc}/100<br>"
                f"Pop. vulnérable: {vp:,}<br>"
                f"Niveau: {lvl}<br>"
                f"<extra></extra>"
            ),
        ))

    fig.update_layout(
        mapbox=dict(style="carto-darkmatter", center=dict(lat=32.55, lon=-6.2), zoom=8),
        height=480, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(bgcolor="rgba(0,0,0,0.6)", font_color="white"),
    )
    return fig
