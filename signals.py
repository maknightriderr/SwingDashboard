"""
signals.py v10 — Production-Grade Signal Engine
Features: Nifty 500 CSV, Liquidity Gate, Sensex, Risk Metrics, Confidence.
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from scipy.signal import find_peaks
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ==============================================================================
# 1. COMPREHENSIVE NSE SWING TRADING WATCHLIST (NIFTY 500 INTEGRATION)
# ==============================================================================

SECTOR_STOCKS = {}
SECTOR_MAP = {}

CSV_PATH = "ind_nifty500list.csv"

if os.path.exists(CSV_PATH):
    try:
        nifty_df = pd.read_csv(CSV_PATH)
        for _, row in nifty_df.iterrows():
            sym = str(row["Symbol"]).strip()
            sec = str(row["Industry"]).strip()
            if sec not in SECTOR_STOCKS:
                SECTOR_STOCKS[sec] = []
            SECTOR_STOCKS[sec].append(sym)
            SECTOR_MAP[sym] = sec
    except Exception as e:
        print(f"Error loading Nifty 500 CSV: {e}")
else:
    SECTOR_STOCKS = {
        "Banking": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK"],
        "IT": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM"],
        "Energy": ["RELIANCE", "ONGC", "NTPC", "POWERGRID"]
    }
    for sector, stocks in SECTOR_STOCKS.items():
        for stock in stocks:
            SECTOR_MAP[stock] = sector

def get_sector(symbol: str) -> str:
    clean_symbol = symbol.upper().replace(".NS", "").replace(".BO", "")
    return SECTOR_MAP.get(clean_symbol, "Others")

SECTOR_INDICES = {
    "Financial Services": "^CNXFIN", "Information Technology": "^CNXIT",
    "Healthcare": "^CNXPHARMA", "Automobile and Auto Components": "^CNXAUTO",
    "Oil Gas & Consumable Fuels": "^CNXENERGY", "Metals & Mining": "^CNXMETAL",
    "Fast Moving Consumer Goods": "^CNXFMCG", "Construction": "^CNXINFRA",
    "Realty": "^CNXREALTY"
}

TRACKED_INDICES = {
    "Sensex": "^BSESN",
    "Nifty 50": "^NSEI", "Bank Nifty": "^NSEBANK",
    "Nifty IT": "^CNXIT", "Nifty Fin": "^CNXFIN",
    "India VIX": "^INDIAVIX",
}

# ─── Robust Data Fetcher (Concurrent) ─────────────────────────────────────────
def _fetch_history(ticker, period="1y", interval="1d"):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval, auto_adjust=False)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        result = pd.DataFrame()
        result["Open"] = df["Open"] if "Open" in df.columns else df["Close"]
        result["Close"] = df["Close"]
        result["High"] = df["High"] if "High" in df.columns else df["Close"]
        result["Low"] = df["Low"] if "Low" in df.columns else df["Close"]
        result["Volume"] = df["Volume"] if "Volume" in df.columns else 0
        result = result.dropna(subset=["Close"]).ffill().bfill()
        return result if not result.empty else None
    except Exception:
        return None

def _bulk_fetch_history(symbols, period="1y"):
    results = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        def fetch_single(sym):
            fetch_sym = sym if sym.startswith("^") else sym + ".NS"
            df = _fetch_history(fetch_sym, period)
            if df is None and not sym.startswith("^"):
                df = _fetch_history(sym + ".BO", period)
            return sym, df
        future_to_sym = {executor.submit(fetch_single, sym): sym for sym in symbols}
        for future in as_completed(future_to_sym):
            try:
                sym, df = future.result()
                if df is not None:
                    results[sym] = df
            except Exception:
                pass
    return results

# ─── Indicator Cache ──────────────────────────────────────────────────────────
_IND_CACHE = {}
_IND_CACHE_TS = {}
_CACHE_TTL = 900

# ─── Wilder's RSI ─────────────────────────────────────────────────────────────
def compute_rsi_wilder(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0/period, min_periods=period).mean
