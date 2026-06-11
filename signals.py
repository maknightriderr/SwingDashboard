"""
signals.py v10 — Production-Grade Signal Engine
Features: Nifty 500 CSV, ₹1Cr Liquidity Gate, Sensex Tracking, Risk Metrics, Confidence Scoring.
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
            if sec not in SECTOR_STOCKS: SECTOR_STOCKS[sec] = []
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
        for stock in stocks: SECTOR_MAP[stock] = sector

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
        if df is None or df.empty or "Close" not in df.columns: return None
        
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
            if df is None and not sym.startswith("^"): df = _fetch_history(sym + ".BO", period)
            return sym, df
        future_to_sym = {executor.submit(fetch_single, sym): sym for sym in symbols}
        for future in as_completed(future_to_sym):
            sym, df = future.result()
            if df is not None: results[sym] = df
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
    avg_gain = gain.ewm(alpha=1.0/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_rsi(series, period=14):
    rsi = compute_rsi_wilder(series, period)
    val = rsi.iloc[-1]
    return round(float(val), 1) if not pd.isna(val) else None

# ─── Divergence Detection ─────────────────────────────────────────────────────
def detect_rsi_divergence(close, rsi_series, window=40):
    if len(close) < window or len(rsi_series) < window:
        return {"bullish_div": False, "bearish_div": False}
    c = close.iloc[-window:].values
    r = rsi_series.iloc[-window:].values
    
    if HAS_SCIPY:
        troughs, _ = find_peaks(-c, distance=5)
        peaks, _ = find_peaks(c, distance=5)
    else:
        # Fallback simple peak detection
        troughs = np.where((c[1:-1] < c[:-2]) & (c[1:-1] < c[2:]))[0] + 1
        peaks = np.where((c[1:-1] > c[:-2]) & (c[1:-1] > c[2:]))[0] + 1

    bullish, bearish = False, False
    if len(troughs) >= 2:
        if c[troughs[-1]] < c[troughs[-2]] and r[troughs[-1]] > r[troughs[-2]]: bullish = True
    if len(peaks) >= 2:
        if c[peaks[-1]] > c[peaks[-2]] and r[peaks[-1]] < r[peaks[-2]]: bearish = True
    return {"bullish_div": bullish, "bearish_div": bearish}

def detect_macd_divergence(close, macd_line, window=40):
    if len(close) < window or len(macd_line) < window:
        return {"bullish_div": False, "bearish_div": False}
    c = close.iloc[-window:].values
    m = macd_line.iloc[-window:].values

    if HAS_SCIPY:
        troughs, _ = find_peaks(-c, distance=5)
        peaks, _ = find_peaks(c, distance=5)
    else:
        troughs = np.where((c[1:-1] < c[:-2]) & (c[1:-1] < c[2:]))[0] + 1
        peaks = np.where((c[1:-1] > c[:-2]) & (c[1:-1] > c[2:]))[0] + 1

    bullish, bearish = False, False
    if len(troughs) >= 2:
        if c[troughs[-1]] < c[troughs[-2]] and m[troughs[-1]] > m[troughs[-2]]: bullish = True
    if len(peaks) >= 2:
        if c[peaks[-1]] > c[peaks[-2]] and m[peaks[-1]] < m[peaks[-2]]: bearish = True
    return {"bullish_div": bullish, "bearish_div": bearish}

# ─── Chart Pattern Detection ──────────────────────────────────────────────────
def detect_price_patterns(high, low, close, vol, vol_avg):
    patterns = []
    if len(close) < 30: return patterns
    cmp = float(close.iloc[-1])
    recent_highs = high.iloc[-16:-1]
    recent_lows = low.iloc[-16:-1]
    range_pct = (recent_highs.max() - recent_lows.min()) / recent_lows.min()
    if range_pct < 0.08:
        if cmp > recent_highs.max() and float(vol.iloc[-1]) > vol_avg * 2: patterns.append("🚀 Vol Breakout")
    pole = close.iloc[-25:-8]
    flag = close.iloc[-8:-1]
    pole_gain = (pole.max() - pole.min()) / pole.min()
    flag_drop = (flag.max() - flag.min()) / flag.max()
    if pole_gain > 0.12 and flag_drop < 0.07 and flag.iloc[-1] < pole.max():
        if cmp > flag.max() and float(vol.iloc[-1]) > vol_avg * 1.5: patterns.append("🚩 Bull Flag Breakout")
        
    if HAS_SCIPY:
        c_vals = close.values
        troughs, _ = find_peaks(-c_vals, distance=10)
        peaks, _ = find_peaks(c_vals, distance=10)
    else:
        c_vals = close.values
        troughs = np.where((c_vals[1:-1] < c_vals[:-2]) & (c_vals[1:-1] < c_vals[2:]))[0] + 1
        peaks = np.where((c_vals[1:-1] > c_vals[:-2]) & (c_vals[1:-1] > c_vals[2:]))[0] + 1

    if len(troughs) >= 2:
        t1, t2 = troughs[-2], troughs[-1]
        p1, p2 = c_vals[t1], c_vals[t2]
        if abs(p1 - p2) / p1 < 0.03 and p2 * 1.01 < cmp < p2 * 1.08: patterns.append("📉 Double Bottom")
    if len(peaks) >= 2:
        p1_idx, p2_idx = peaks[-2], peaks[-1]
        p1, p2 = c_vals[p1_idx], c_vals[p2_idx]
        if abs(p1 - p2) / p1 < 0.03 and p2 * 0.92 < cmp < p2 * 0.99: patterns.append("📈 Double Top")
    if len(peaks) >= 3:
        p1, p2, p3 = c_vals[peaks[-3]], c_vals[peaks[-2]], c_vals[peaks[-1]]
        if p2 > p1 and p2 > p3 and abs(p1 - p3) / p1 < 0.05:
            if cmp < p3 * 0.98: patterns.append("🏔️ Head & Shoulders (Top)")
    if len(troughs) >= 3:
        t1, t2, t3 = c_vals[troughs[-3]], c_vals[troughs[-2]], c_vals[troughs[-1]]
        if t2 < t1 and t2 < t3 and abs(t1 - t3) / t1 < 0.05:
            if cmp > t3 * 1.02: patterns.append("🛤️ Inverse H&S (Bottom)")
    return patterns

# ─── Candlestick Confirmation ─────────────────────────────────────────────────
def detect_candlesticks(open_p, high, low, close):
    candles = []
    if len(close) < 5: return candles
    o, h, l, c = float(open_p.iloc[-1]), float(high.iloc[-1]), float(low.iloc[-1]), float(close.iloc[-1])
    po, pc = float(open_p.iloc[-2]), float(close.iloc[-2])
    body = abs(c - o)
    total_range = h - l
    if total_range == 0: return candles
    lower_wick = o - l if c > o else c - l
    upper_wick = h - c if c > o else h - o
    if lower_wick > (body * 2) and upper_wick < (body * 0.5) and body > (total_range * 0.1): candles.append("🔨 Bullish Hammer")
    if pc < po and c > o:
        if o <= pc and c >= po and body > abs(pc - po): candles.append("🟩 Bullish Engulfing")
    if pc > po and c < o:
        if o >= pc and c <= po and body > abs(pc - po): candles.append("🟥 Bearish Engulfing")
    return candles

# ─── Market Regime Detection ──────────────────────────────────────────────────
_market_regime_cache = {"ts": 0, "data": None}

def get_market_regime():
    now = time.time()
    if _market_regime_cache["data"] and (now - _market_regime_cache["ts"]) < _CACHE_TTL:
        return _market_regime_cache["data"]

    indices_data = {}
    bulk_data = _bulk_fetch_history(list(TRACKED_INDICES.values()), period="1y")
    
    for name, symbol in TRACKED_INDICES.items():
        if symbol in bulk_data:
            df = bulk_data[symbol]
            if df is not None and len(df) >= 2:
                current = float(df["Close"].iloc[-1])
                prev = float(df["Close"].iloc[-2])
                chg = round((current / prev - 1) * 100, 2)
                indices_data[name] = {"price": round(current, 2), "chg_pct": chg}
            elif df is not None and len(df) == 1:
                current = float(df["Close"].iloc[-1])
                indices_data[name] = {"price": round(current, 2), "chg_pct": 0.0}

    nifty = indices_data.get("Nifty 50", {})
    nifty_close = nifty.get("price")
    regime, trend, nifty_rsi = "Unknown", "Sideways", None
    support, resistance, conf = None, None, 50

    if nifty_close and "^NSEI" in bulk_data:
        df = bulk_data["^NSEI"]
        if len(df) >= 50:
            close = df["Close"]
            ema20 = float(close.ewm(span=20).mean().iloc[-1])
            ema50 = float(close.ewm(span=50).mean().iloc[-1])
            ema200 = float(close.ewm(span=200).mean().iloc[-1]) if len(close) >= 200 else None
            nifty_rsi = compute_rsi(close)
            support = float(df["Low"].rolling(20).min().iloc[-1])
            resistance = float(df["High"].rolling(20).max().iloc[-1])

            if ema200 and nifty_close > ema200:
                if nifty_close > ema20 > ema50: regime, trend = "Strong Bull", "Uptrend"
                elif nifty_close > ema20: regime, trend = "Bull", "Uptrend"
                else: regime, trend = "Bull Pullback", "Pullback"
            elif ema200 and nifty_close < ema200:
                if nifty_close < ema20 < ema50: regime, trend = "Strong Bear", "Downtrend"
                elif nifty_close < ema20: regime, trend = "Bear", "Downtrend"
                else: regime, trend = "Bear Rally", "Relief Rally"
            else: regime, trend = "Neutral", "Sideways"

            if regime == "Strong Bull": conf = 85
            elif regime == "Bull": conf = 70
            elif regime == "Bull Pullback": conf = 55
            elif regime == "Strong Bear": conf = 85
            elif regime == "Bear": conf = 70
            elif regime == "Bear Rally": conf =
