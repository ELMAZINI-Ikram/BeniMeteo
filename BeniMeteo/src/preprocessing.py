"""
preprocessing.py — Nettoyage et enrichissement des données climatiques.
"""

import pandas as pd
import numpy as np


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage complet + variables dérivées."""
    df = df.copy()
    df = df.ffill().bfill()

    if "tmax" in df.columns and "tmin" in df.columns:
        mask = df["tmax"] < df["tmin"]
        df.loc[mask, ["tmax", "tmin"]] = df.loc[mask, ["tmin", "tmax"]].values
        df["temp_mean"]  = (df["tmax"] + df["tmin"]) / 2
        df["temp_range"] = df["tmax"] - df["tmin"]

    if "tmean" in df.columns:
        df["temp_mean"] = df["tmean"]

    if "humidity_max" in df.columns and "humidity_min" in df.columns:
        df["humidity_mean"] = (df["humidity_max"] + df["humidity_min"]) / 2

    # Heat Index
    if "temp_mean" in df.columns and "humidity_mean" in df.columns:
        T = df["temp_mean"]
        H = df["humidity_mean"]
        df["heat_index"] = (
            -8.78469475556
            + 1.61139411 * T
            + 2.33854883889 * H
            - 0.14611605 * T * H
            - 0.012308094 * T**2
            - 0.0164248277778 * H**2
            + 0.002211732 * T**2 * H
            + 0.00072546 * T * H**2
            - 0.000003582 * T**2 * H**2
        ).round(1)

    if "sunshine_sec" in df.columns:
        df["sunshine_h"] = (df["sunshine_sec"] / 3600).round(2)

    df["month"]       = df.index.month
    df["year"]        = df.index.year
    df["day_of_year"] = df.index.dayofyear
    df["season"]      = df["month"].map({
        12: "Hiver", 1: "Hiver", 2: "Hiver",
        3: "Printemps", 4: "Printemps", 5: "Printemps",
        6: "Été", 7: "Été", 8: "Été",
        9: "Automne", 10: "Automne", 11: "Automne",
    })

    if "temp_mean" in df.columns:
        monthly_ref       = df.groupby("month")["temp_mean"].transform("mean")
        df["temp_anomaly"] = (df["temp_mean"] - monthly_ref).round(2)

    if "tmax" in df.columns:
        df["canicule"] = (df["tmax"] >= 40).astype(int)
        df["chaleur"]  = (df["tmax"] >= 35).astype(int)

    if "tmin" in df.columns:
        df["gel"] = (df["tmin"] <= 0).astype(int)

    if "precip" in df.columns:
        df["precip_7d"]  = df["precip"].rolling(7, min_periods=1).sum().round(1)
        df["precip_30d"] = df["precip"].rolling(30, min_periods=1).sum().round(1)
        df["jour_pluie"] = (df["precip"] > 0.1).astype(int)

    # Lags pour ML
    if "temp_mean" in df.columns:
        for lag in [1, 3, 7, 14, 30]:
            df[f"tmean_lag{lag}"] = df["temp_mean"].shift(lag)

    return df


def get_monthly_stats(df: pd.DataFrame) -> pd.DataFrame:
    agg = {k: v for k, v in {
        "tmax": ["mean", "max"], "tmin": ["mean", "min"],
        "temp_mean": ["mean"], "precip": ["sum", "mean"],
        "humidity_mean": ["mean"], "wind_max": ["mean", "max"],
        "uv_max": ["mean", "max"], "canicule": ["sum"], "jour_pluie": ["sum"],
    }.items() if k in df.columns}
    monthly = df.groupby(["year", "month"]).agg(agg)
    monthly.columns = ["_".join(c) for c in monthly.columns]
    return monthly.reset_index()


def get_annual_stats(df: pd.DataFrame) -> pd.DataFrame:
    agg = {k: v for k, v in {
        "tmax": ["mean", "max"], "tmin": ["mean", "min"],
        "temp_mean": ["mean"], "precip": ["sum"],
        "canicule": ["sum"], "gel": ["sum"], "jour_pluie": ["sum"],
    }.items() if k in df.columns}
    annual = df.groupby("year").agg(agg)
    annual.columns = ["_".join(c) for c in annual.columns]
    return annual.reset_index()
