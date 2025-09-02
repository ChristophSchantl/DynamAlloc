# streamlit_app.py
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import math
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Quant-Portfolio: China • Rohstoffe • Gold • EM", layout="wide")

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_prices(tickers: List[str], start: str) -> pd.DataFrame:
    df = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.sort_index().ffill().dropna(how="all", axis=1)
    return df

@st.cache_data(show_spinner=False)
def get_long_names(tickers: List[str]) -> dict:
    names = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            names[t] = info.get("longName", t)
        except Exception:
            names[t] = t
    return names

def ann_factor(freq: str) -> float:
    return {"D":252, "W":52, "M":12}.get(freq, 252)

def cagr(series: pd.Series, freq: str="D") -> float:
    if len(series) < 2: return float("nan")
    years = len(series)/ann_factor(freq)
    return series.iloc[-1]**(1/years)-1

def safe_div(a, b):
    return a / b if b != 0 else np.nan

def rename_with_long(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    return df.rename(index=mapping, columns=mapping)

# ─────────────────────────────────────────────────────────────
# Sidebar – Parameter
# ─────────────────────────────────────────────────────────────
st.sidebar.header("Parameter")
start_date = st.sidebar.date_input("Startdatum", value=datetime(2015,1,1))
rebalance = st.sidebar.selectbox("Rebalancing", ["täglich","wöchentlich","monatlich"], index=2)
ma_fast = st.sidebar.slider("UUP Fast MA", 10, 100, 50, step=5)
ma_slow = st.sidebar.slider("UUP Slow MA", 100, 300, 200, step=10)
vix_thr  = st.sidebar.slider("VIXY Schwelle", 10.0, 40.0, 20.0, step=0.5)
gold_over_in_stress = st.sidebar.slider("Gold-Übergewicht bei Stress (pp)", 0.0, 0.20, 0.05, step=0.01)
usd_weak_tilt = st.sidebar.slider("Tilt bei schwachem USD (pp)", 0.0, 0.30, 0.15, step=0.01)
tc_bps = st.sidebar.slider("Transaktionskosten (bps pro Trade)", 0, 50, 5, step=1)

st.sidebar.markdown("---")
st.sidebar.caption("Signale aus UUP (USD) und VIXY. Daten via Yahoo Finance.")

# ─────────────────────────────────────────────────────────────
# Universum
# ─────────────────────────────────────────────────────────────
universe: Dict[str, List[str]] = {
    "China": ["MCHI","ASHR"],
    "Rohstoffe": ["COPX","LIT","DBB"],
    "Öl": ["XLE"],
    "Gold": ["GLD","GDX"],
    "EM": ["EWZ","EWW"],
    "Signals": ["UUP","VIXY"]
}
tickers_all = sum(universe.values(), [])

# Basisgewichte (neutral)
base_w = pd.Series({
    "MCHI":0.10,"ASHR":0.10,
    "COPX":0.10,"LIT":0.10,"DBB":0.05,
    "XLE":0.15,
    "GLD":0.15,"GDX":0.10,
    "EWZ":0.10,"EWW":0.05
})

# ─────────────────────────────────────────────────────────────
# Daten laden
# ─────────────────────────────────────────────────────────────
prices = load_prices(tickers_all, start=start_date.strftime("%Y-%m-%d"))
long_names = get_long_names(list(prices.columns))

missing = [t for t in base_w.index if t not in prices.columns]
if missing:
    st.warning(f"Keine Daten für: {', '.join(missing)}")
    base_w = base_w.drop(index=missing)

# Returns
rets = prices.pct_change().dropna()
# Signale
uup = prices.get("UUP")
vixy = prices.get("VIXY")
if uup is None or vixy is None:
    st.error("Signal-Ticker UUP und/oder VIXY fehlen. Bitte prüfen.")
    st.stop()

uup_ma_fast = uup.rolling(ma_fast).mean()
uup_ma_slow = uup.rolling(ma_slow).mean()
signal_usd_weak = (uup_ma_fast < uup_ma_slow).astype(int)

signal_high_vol = (vixy > vix_thr).astype(int)
signals = pd.DataFrame({
    "USD_weak": signal_usd_weak,
    "HighVol": signal_high_vol
}).dropna()

# ─────────────────────────────────────────────────────────────
# Dynamische Gewichte pro Tag
# ─────────────────────────────────────────────────────────────
def make_weights_row(date) -> pd.Series:
    w = base_w.copy()
    if signals.at[date, "USD_weak"] == 1:
        add = usd_weak_tilt
        inc = add/4
        for t in ["MCHI","ASHR","COPX","LIT"]:
            if t in w: w[t] += inc
        if "GLD" in w: w["GLD"] = max(0, w["GLD"] - add/2)
        if "GDX" in w: w["GDX"] = max(0, w["GDX"] - add/2)

    if signals.at[date, "HighVol"] == 1:
        add_g = gold_over_in_stress
        if "GLD" in w: w["GLD"] += add_g
        cut = add_g
        for t in ["MCHI","ASHR","EWZ","EWW","COPX","LIT","DBB","XLE","GDX"]:
            if t in w and cut > 0:
                delta = min(w[t], cut/9)
                w[t] -= delta

    w = w.clip(lower=0)
    w = w / w.sum()
    return w

weights_ts = pd.DataFrame({d: make_weights_row(d) for d in signals.index}).T.reindex(rets.index).ffill()

# ─────────────────────────────────────────────────────────────
# Rebalancing + Transaktionskosten
# ─────────────────────────────────────────────────────────────
if rebalance == "täglich":
    rb_dates = rets.index
elif rebalance == "wöchentlich":
    rb_dates = rets.index[rets.index.weekday == 0]
else:
    rb_dates = rets.index[rets.index.is_month_start]

weights_rb = weights_ts.copy()
mask = ~weights_rb.index.isin(rb_dates)
weights_rb[mask] = np.nan
weights_rb = weights_rb.ffill()

# Portfolio-Return vor Kosten
port_ret_gross = (rets[weights_rb.columns] * weights_rb.shift()).sum(axis=1)

# Turnover
w_prev = weights_rb.shift().fillna(method="bfill").fillna(0)
turnover = (weights_rb - w_prev).abs().sum(axis=1).where(weights_rb.index.isin(rb_dates), 0.0)
tc = turnover * (tc_bps/10000.0)
port_ret_net = port_ret_gross - tc

equity_curve = (1 + port_ret_net.fillna(0)).cumprod()

# ─────────────────────────────────────────────────────────────
# Kennzahlen
# ─────────────────────────────────────────────────────────────
ann_ret = port_ret_net.mean() * ann_factor("D")
ann_vol = port_ret_net.std() * math.sqrt(252)
sharpe = safe_div(ann_ret, ann_vol)
max_dd = (equity_curve / equity_curve.cummax() - 1).min()

# Contributions
contrib = (rets[weights_rb.columns] * weights_rb.shift()).sum().sort_values(ascending=False)

# ─────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────
colA, colB, colC, colD = st.columns(4)
colA.metric("CAGR", f"{cagr(equity_curve, 'D')*100:,.2f}%")
colB.metric("Vol (ann.)", f"{ann_vol*100:,.2f}%")
colC.metric("Sharpe (rf=0)", f"{sharpe:,.2f}")
colD.metric("Max Drawdown", f"{max_dd*100:,.2f}%")

st.plotly_chart(
    go.Figure().add_trace(
        go.Scatter(x=equity_curve.index, y=equity_curve.values, name="Equity")
    ).update_layout(title="Kumulierte Wertentwicklung", xaxis_title="", yaxis_title="Start=1"),
    use_container_width=True
)

# Aktuelle Gewichte mit Langnamen
last_w = weights_rb.iloc[-1].sort_values(ascending=False)
st.subheader("Aktuelle Gewichte")
st.dataframe(
    rename_with_long(last_w.to_frame("Gewicht"), long_names).style.format("{:.2%}")
)

# Contributions mit Langnamen
st.subheader("Beitragsanalyse seit Start")
st.dataframe(
    rename_with_long(contrib.to_frame("Beitrag (Summe)"), long_names).style.format("{:.4f}")
)

# Downloads – EU-Format
st.subheader("Downloads (EU-Format)")
csv_eq = equity_curve.to_frame("equity").to_csv(index=True, sep=";", decimal=",").encode("utf-8-sig")
st.download_button("Equity-Kurve (CSV)", data=csv_eq, file_name="equity_curve_eu.csv", mime="text/csv")

csv_w = weights_rb.to_csv(index=True, sep=";", decimal=",").encode("utf-8-sig")
st.download_button("Gewichte über Zeit (CSV)", data=csv_w, file_name="weights_timeseries_eu.csv", mime="text/csv")

csv_ret = port_ret_net.to_frame("ret").to_csv(index=True, sep=";", decimal=",").encode("utf-8-sig")
st.download_button("Tägliche Returns (CSV)", data=csv_ret, file_name="daily_returns_eu.csv", mime="text/csv")

st.caption("Hinweis: CSV-Exports sind im EU-Format (Semikolon, Komma als Dezimal). UUP/VIXY sind US-ETFs.")
