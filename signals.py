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
from scipy.signal import find_peaks

# ==============================================================================
# 1. COMPREHENSIVE NSE SWING TRADING WATCHLIST (NIFTY 500 INTEGRATION)
# ==============================================================================

SECTOR_STOCKS = {}
SECTOR_MAP = {}

# We dynamically load the Nifty 500 from the official NSE CSV.
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
    # Fallback to the Top F&O list if the CSV is missing
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

# Maintain the core indices for macro market detection
SECTOR_INDICES = {
    "Financial Services": "^CNXFIN", "Information Technology": "^CNXIT", 
    "Healthcare": "^CNXPHARMA", "Automobile and Auto Components": "^CNXAUTO", 
    "Oil Gas & Consumable Fuels": "^CNXENERGY", "Metals & Mining": "^CNXMETAL", 
    "Fast Moving Consumer Goods": "^CNXFMCG", "Construction": "^CNXINFRA", 
    "Realty": "^CNXREALTY", "Media": "^CNXMEDIA",
    "Consumer Durables": "^CNXCONSUM",
    "Consumer Services": "^CNXSERVICE",
    "PSU Bank": "^CNXPSUBANK",
    "Private Bank": "^CNXPVTBANK",
    "Bank": "^NSEBANK"
}

TRACKED_INDICES = {
    "Sensex": "^BSESN", 
    "Nifty 50": "^NSEI", 
    "Nifty Midcap": "^NSEMDCP50",  # Tracks the broader mid-tier momentum
    "Nifty Smallcap": "^CNXSC",   # Tracks aggressive high-beta retail momentum
    "Bank Nifty": "^NSEBANK",
    "Nifty IT": "^CNXIT", 
    "India VIX": "^INDIAVIX"      # Core volatility tracker
}

# ─── Robust Data Fetcher (Concurrent) ─────────────────────────────────────────
def _fetch_history(ticker, period="1y", interval="1d"):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval, auto_adjust=True)
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

def sanitize_ticker(sym):
    """Strips existing extensions to prevent SNOWMAN.NS.NS"""
    clean = str(sym).upper().strip()
    for suffix in [".NS", ".BO", ".NSE", ".BSE"]:
        if clean.endswith(suffix):
            clean = clean[:-len(suffix)]
    return clean

def _bulk_fetch_history(symbols, period="1y"):
    results = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        def fetch_single(sym):
            if sym.startswith("^"):
                df = _fetch_history(sym, period)
                return sym, df
                
            clean_sym = sanitize_ticker(sym)
            df = _fetch_history(clean_sym + ".NS", period)
            if df is None:
                df = _fetch_history(clean_sym + ".BO", period)
            return sym, df

        future_to_sym = {executor.submit(fetch_single, sym): sym for sym in symbols}
        for future in as_completed(future_to_sym):
            sym, df = future.result()
            if df is not None:
                results[sym] = df
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

# ─── Divergence Detection (Scipy Peaks) ───────────────────────────────────────
def detect_rsi_divergence(close, rsi_series, window=40):
    if len(close) < window or len(rsi_series) < window:
        return {"bullish_div": False, "bearish_div": False}
    c = close.iloc[-window:].values
    r = rsi_series.iloc[-window:].values
    troughs, _ = find_peaks(-c, distance=5)
    peaks, _ = find_peaks(c, distance=5)
    
    bullish, bearish = False, False
    if len(troughs) >= 2:
        if c[troughs[-1]] < c[troughs[-2]] and r[troughs[-1]] > r[troughs[-2]]:
            bullish = True
    if len(peaks) >= 2:
        if c[peaks[-1]] > c[peaks[-2]] and r[peaks[-1]] < r[peaks[-2]]:
            bearish = True
    return {"bullish_div": bullish, "bearish_div": bearish}

def detect_macd_divergence(close, macd_line, window=40):
    if len(close) < window or len(macd_line) < window:
        return {"bullish_div": False, "bearish_div": False}
    c = close.iloc[-window:].values
    m = macd_line.iloc[-window:].values
    troughs, _ = find_peaks(-c, distance=5)
    peaks, _ = find_peaks(c, distance=5)
    
    bullish, bearish = False, False
    if len(troughs) >= 2:
        if c[troughs[-1]] < c[troughs[-2]] and m[troughs[-1]] > m[troughs[-2]]:
            bullish = True
    if len(peaks) >= 2:
        if c[peaks[-1]] > c[peaks[-2]] and m[peaks[-1]] < m[peaks[-2]]:
            bearish = True
    return {"bullish_div": bullish, "bearish_div": bearish}

# ─── UPGRADED CHART PATTERN DETECTION ────────────────────────────────────────
# Fixes: double bottom tolerance, H&S neckline break, adds Cup & Handle

def detect_price_patterns(high, low, close, vol, vol_avg):
    patterns = []
    if len(close) < 30:
        return patterns

    cmp = float(close.iloc[-1])
    c_vals = close.values
    
    troughs, _ = find_peaks(-c_vals, distance=8, prominence=c_vals.std() * 0.3)
    peaks, _ = find_peaks(c_vals, distance=8, prominence=c_vals.std() * 0.3)

    # Volume Breakout
    if len(close) >= 20:
        recent_h = high.iloc[-20:-1].max()
        if cmp > recent_h and float(vol.iloc[-1]) > vol_avg * 2.5:
            patterns.append("🚀 Vol Breakout")

    # Bull Flag
    if len(close) >= 30:
        pole = close.iloc[-30:-10]
        flag = close.iloc[-10:-1]
        p_gain = (pole.max() - pole.min()) / (pole.min() + 1e-8)
        if p_gain > 0.08 and flag.iloc[-1] < pole.max():
            if cmp > flag.max() and float(vol.iloc[-1]) > vol_avg * 2.0:
                patterns.append("🚩 Bull Flag Breakout")
    
    return patterns

    # ── Double Bottom — relaxed to 8% tolerance + volume expansion ───────────
    if len(troughs) >= 2:
        t1, t2 = troughs[-2], troughs[-1]
        p1, p2 = c_vals[t1], c_vals[t2]
        depth_ok    = abs(p1 - p2) / (p1 + 1e-8) < 0.08   # was 0.03 — now 8%
        price_ok    = p2 * 1.00 < cmp < p2 * 1.12           # breaking up from 2nd trough
        vol_confirm = float(vol.iloc[-1]) > vol_avg * 1.2   # volume expansion
        if depth_ok and price_ok and vol_confirm:
            patterns.append("📉 Double Bottom")

    # ── Double Top — same relaxation ─────────────────────────────────────────
    if len(peaks) >= 2:
        p1_idx, p2_idx = peaks[-2], peaks[-1]
        v1, v2 = c_vals[p1_idx], c_vals[p2_idx]
        if abs(v1 - v2) / (v1 + 1e-8) < 0.08 and v2 * 0.88 < cmp < v2 * 0.99:
            patterns.append("📈 Double Top")

    # ── Head & Shoulders with neckline break confirmation ────────────────────
    if len(peaks) >= 3 and len(troughs) >= 2:
        p1, p2, p3 = c_vals[peaks[-3]], c_vals[peaks[-2]], c_vals[peaks[-1]]
        head_valid = p2 > p1 and p2 > p3 and abs(p1 - p3) / (p1 + 1e-8) < 0.06
        if head_valid:
            # Neckline = average of the two troughs between shoulders
            neckline = (c_vals[troughs[-2]] + c_vals[troughs[-1]]) / 2
            # Signal only fires when price breaks below neckline on volume
            if cmp < neckline * 0.99 and float(vol.iloc[-1]) > vol_avg * 1.3:
                patterns.append("🏔️ Head & Shoulders (Top)")

    # ── Inverse H&S ───────────────────────────────────────────────────────────
    if len(troughs) >= 3 and len(peaks) >= 2:
        t1, t2, t3 = c_vals[troughs[-3]], c_vals[troughs[-2]], c_vals[troughs[-1]]
        head_valid = t2 < t1 and t2 < t3 and abs(t1 - t3) / (t1 + 1e-8) < 0.06
        if head_valid:
            neckline = (c_vals[peaks[-2]] + c_vals[peaks[-1]]) / 2
            if cmp > neckline * 1.01 and float(vol.iloc[-1]) > vol_avg * 1.3:
                patterns.append("🛤️ Inverse H&S (Bottom)")

    # ── Cup & Handle ─────────────────────────────────────────────────────────
    # Needs at least 60 bars: cup forms over 30-50 bars, handle over 5-15
    if len(close) >= 60:
        cup_window = close.iloc[-60:-10]
        handle     = close.iloc[-10:]
        cup_left   = float(cup_window.iloc[0])
        cup_right  = float(cup_window.iloc[-1])
        cup_base   = float(cup_window.min())

        cup_depth  = (cup_left - cup_base) / (cup_left + 1e-8)
        rim_match  = abs(cup_left - cup_right) / (cup_left + 1e-8)
        handle_ret = (float(handle.max()) - float(handle.min())) / (float(handle.max()) + 1e-8)
        breakout   = cmp > float(handle.max()) * 0.995

        if (0.10 < cup_depth < 0.40 and     # cup is 10-40% deep
                rim_match < 0.06 and         # both rims near equal height
                handle_ret < 0.08 and        # handle retracement < 8%
                breakout and                 # breaking above handle high
                float(vol.iloc[-1]) > vol_avg * 1.5):
            patterns.append("☕ Cup & Handle Breakout")

    return patterns

# ─── Candlestick Confirmation ─────────────────────────────────────────────────
def detect_candlesticks(open_p, high, low, close):
    candles = []
    if len(close) < 5:
        return candles

    def _candle(i):
        o, h, l, c = float(open_p.iloc[i]), float(high.iloc[i]), float(low.iloc[i]), float(close.iloc[i])
        body      = abs(c - o)
        rng       = h - l
        if rng < 1e-8:
            return None
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        bullish    = c > o
        return dict(o=o, h=h, l=l, c=c, body=body, rng=rng,
                    upper_wick=upper_wick, lower_wick=lower_wick, bullish=bullish)

    c0 = _candle(-1)   # today
    c1 = _candle(-2)   # yesterday
    c2 = _candle(-3) if len(close) >= 3 else None  # day before

    if not c0 or not c1:
        return candles

    # ── Single-candle patterns ────────────────────────────────────────────────

    # Hammer / Inverted Hammer — body in upper third, long lower wick
    if (c0["lower_wick"] >= c0["rng"] * 0.55 and       # lower wick ≥ 55% of range
            c0["upper_wick"] <= c0["rng"] * 0.15 and   # small upper wick
            c0["body"]       >= c0["rng"] * 0.05):      # real body (not doji)
        candles.append("🔨 Bullish Hammer")

    # Shooting Star — body in lower third, long upper wick (bearish)
    if (c0["upper_wick"] >= c0["rng"] * 0.55 and
            c0["lower_wick"] <= c0["rng"] * 0.15 and
            c0["body"]       >= c0["rng"] * 0.05 and
            not c0["bullish"]):
        candles.append("💫 Shooting Star")

    # Doji — almost no body, uncertainty
    if c0["body"] <= c0["rng"] * 0.07:
        candles.append("〰️ Doji (Indecision)")

    # ── Two-candle patterns ───────────────────────────────────────────────────

    # Bullish Engulfing — strict: prev bearish, curr bullish, curr body wraps prev body
    if (not c1["bullish"] and c0["bullish"] and         # color flip required
            c0["o"] <= c1["c"] and                      # open below prev close
            c0["c"] >= c1["o"] and                      # close above prev open
            c0["body"] > c1["body"] * 1.0):             # body must be larger
        candles.append("🟩 Bullish Engulfing")

    # Bearish Engulfing — strict: prev bullish, curr bearish
    if (c1["bullish"] and not c0["bullish"] and
            c0["o"] >= c1["c"] and
            c0["c"] <= c1["o"] and
            c0["body"] > c1["body"] * 1.0):
        candles.append("🟥 Bearish Engulfing")

    # Bullish Harami — small bullish inside large bearish
    if (not c1["bullish"] and c0["bullish"] and
            c0["o"] > c1["c"] and c0["c"] < c1["o"] and
            c0["body"] < c1["body"] * 0.5):
        candles.append("🟢 Bullish Harami")

    # Piercing Line — bullish reversal after gap-down open, closes above midpoint
    if (not c1["bullish"] and c0["bullish"] and
            c0["o"] < c1["l"] and                       # gaps below prev low
            c0["c"] > (c1["o"] + c1["c"]) / 2 and      # closes above midpoint
            c0["c"] < c1["o"]):                         # doesn't fully engulf
        candles.append("🔆 Piercing Line")

    # ── Three-candle patterns ─────────────────────────────────────────────────

    if c2:
        # Morning Star — strong bullish reversal
        #   Day1: large bearish, Day2: small body gap down, Day3: large bullish > 50% of Day1
        if (not c2["bullish"] and
                c2["body"] >= c2["rng"] * 0.5 and
                c1["body"] <= c1["rng"] * 0.3 and       # small indecision
                c0["bullish"] and
                c0["body"] >= c0["rng"] * 0.5 and
                c0["c"] > (c2["o"] + c2["c"]) / 2):     # recovers into Day1 body
            candles.append("🌅 Morning Star")

        # Evening Star — strong bearish reversal
        if (c2["bullish"] and
                c2["body"] >= c2["rng"] * 0.5 and
                c1["body"] <= c1["rng"] * 0.3 and
                not c0["bullish"] and
                c0["body"] >= c0["rng"] * 0.5 and
                c0["c"] < (c2["o"] + c2["c"]) / 2):
            candles.append("🌆 Evening Star")

        # Three White Soldiers — sustained buying pressure
        if (c2["bullish"] and c1["bullish"] and c0["bullish"] and
                c1["o"] > c2["o"] and c0["o"] > c1["o"] and   # each opens higher
                c1["c"] > c2["c"] and c0["c"] > c1["c"] and   # each closes higher
                c0["body"] >= c0["rng"] * 0.5 and
                c1["body"] >= c1["rng"] * 0.5):
            candles.append("🪖 Three White Soldiers")

        # Three Black Crows — sustained selling
        if (not c2["bullish"] and not c1["bullish"] and not c0["bullish"] and
                c1["o"] < c2["o"] and c0["o"] < c1["o"] and
                c1["c"] < c2["c"] and c0["c"] < c1["c"] and
                c0["body"] >= c0["rng"] * 0.5 and
                c1["body"] >= c1["rng"] * 0.5):
            candles.append("🦅 Three Black Crows")

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
            
            # Calculate Major Support & Resistance Zones
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

            # Institutional Confidence Scoring
            if regime == "Strong Bull": conf = 85
            elif regime == "Bull": conf = 70
            elif regime == "Bull Pullback": conf = 55
            elif regime == "Strong Bear": conf = 85
            elif regime == "Bear": conf = 70
            elif regime == "Bear Rally": conf = 55
            
            # --- INSTITUTIONAL CONFIDENCE SCORING (MOMENTUM REWARDED) ---
    if nifty_rsi:
        if nifty_rsi > 70: 
            conf = min(95, conf + 15)   # Reward Breakouts! (Power Zone)
        elif nifty_rsi < 40: 
            conf = max(20, conf - 20)   # Penalize bleeding/crashing markets

    # --- MOMENTUM RISK LABELS ---
    risk = "Neutral"
    if nifty_rsi:
        if nifty_rsi > 70: risk = "High Momentum (Power Zone)"
        elif nifty_rsi > 60: risk = "Building Momentum"
        elif nifty_rsi < 40: risk = "High Risk (Downtrend/Bleeding)"

    result = {
        "regime": regime, "trend": trend,
        "nifty_close": nifty_close, "nifty_rsi": nifty_rsi,
        "risk_level": risk, "indices": indices_data,
        "support": support, "resistance": resistance,
        "confidence": conf
    }
    _market_regime_cache["data"] = result
    _market_regime_cache["ts"] = now
    return result

# ─── Technical Indicators ─────────────────────────────────────────────────────
def _compute_indicators_raw(symbol, period="1y", prefetched_df=None):
    # --- 1. RESOLVE TICKER SUFFIXES IMMEDIATELY ---
    # Strip any existing suffix first to avoid .NS.NS
    base_symbol = symbol.upper().split('.')[0].strip()
    tickers_to_try = [f"{base_symbol}.NS", f"{base_symbol}.BO"]
    
    df = prefetched_df
    if df is None:
        for ticker in tickers_to_try:
            df = _fetch_history(ticker, period=period, interval="1d")
            if df is not None: 
                break

    if df is None or len(df) < 50:
        return None

    open_p = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    vol = df["Volume"]

    cmp = float(close.iloc[-1])
    if pd.isna(cmp) or cmp <= 0:
        return None

    # --- 2. INDICATORS (FAST MOMENTUM EMAs) ---
    rsi_series = compute_rsi_wilder(close, 14)
    rsi = round(float(rsi_series.iloc[-1]), 1) if not pd.isna(rsi_series.iloc[-1]) else None

    ema9 = float(close.ewm(span=9).mean().iloc[-1])
    ema21 = float(close.ewm(span=21).mean().iloc[-1])
    ema50 = float(close.ewm(span=50).mean().iloc[-1])

    macd_line = close.ewm(span=12).mean() - close.ewm(span=26).mean()
    signal_line = macd_line.ewm(span=9).mean()
    macd_bullish = False
    if len(macd_line) >= 2:
        macd_bullish = (macd_line.iloc[-1] > signal_line.iloc[-1])

    # --- 3. VOLATILITY & VOLUME ---
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else float(high.iloc[-1] - low.iloc[-1])
    vol_avg = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else float(vol.mean())
    vol_ratio = float(vol.iloc[-1]) / vol_avg if vol_avg > 0 else 1.0

    # --- 4. SUPPORT/RESISTANCE (7-DAY MOMENTUM) ---
    support = float(low.rolling(7).min().iloc[-1])
    resistance = float(high.rolling(7).max().iloc[-1])

    # --- 5. FIBONACCI & TREND ---
    trend = "Sideways"
    if cmp > ema9 > ema21: trend = "Strong Uptrend"
    elif cmp < ema9 < ema21: trend = "Strong Downtrend"

    fib_h, fib_l = float(high.tail(60).max()), float(low.tail(60).min())
    fib_d = fib_h - fib_l
    
    # --- 6. RETURN STRUCTURE ---
    return {
        "symbol": base_symbol, 
        "cmp": round(cmp, 2), 
        "rsi": rsi,
        "ema20": round(ema9, 2), 
        "ema50": round(ema21, 2), 
        "ema200": round(ema50, 2),
        "macd_bullish": macd_bullish,
        "atr": round(atr, 2), 
        "vol_ratio": round(vol_ratio, 2),
        "support": round(support, 2), 
        "resistance": round(resistance, 2),
        "trend": trend,
        "fib_382": round(fib_h - fib_d * 0.382, 2),
        "patterns": detect_price_patterns(high, low, close, vol, vol_avg)
    }

def compute_indicators(symbol, period="1y", prefetched_df=None):
    now = time.time()
    key = f"{symbol}_{period}"
    if key in _IND_CACHE and (now - _IND_CACHE_TS.get(key, 0)) < _CACHE_TTL:
        return _IND_CACHE[key]
    result = _compute_indicators_raw(symbol, period, prefetched_df)
    if result is not None:
        _IND_CACHE[key] = result
        _IND_CACHE_TS[key] = now
    return result

# ─── Expert Signal Engine ─────────────────────────────────────────────────────
def generate_signals(trades_df):
    signals = []
    open_trades = trades_df[trades_df["status"] == "Open"].copy()
    if open_trades.empty: return signals

    market = get_market_regime()
    is_bear = market["regime"] in ("Strong Bear", "Bear")

    unique_symbols = open_trades["stock"].unique().tolist()
    bulk_data = _bulk_fetch_history(unique_symbols, period="1y")

    for _, row in open_trades.iterrows():
        symbol, buy_at, qty, tid = row["stock"], row["buy_at"], row["quantity"], row["id"]
        
        df = bulk_data.get(symbol)
        ind = compute_indicators(symbol, period="1y", prefetched_df=df)

        if ind is None:
            signals.append({
                "id": tid, "stock": symbol, "sector": get_sector(symbol),
            "action": action, "reason": final_reason, "strength": strength,
            "cmp": cmp, "rsi": rsi, "ema20": ema20, "ema50": ema50, "atr": atr,
            "pct_from_buy": pct, "buy_at": buy_at, "quantity": qty,
            "target": target, "stop_loss": stop_loss,
            "avg_price": avg_price, "new_avg": new_avg, "new_sl": new_sl,
            "macd_signal": macd_lbl, "bb_position": bb_lbl,
            "trend": trend, "support": support, "resistance": resistance,
            "risk_reward": rr, "vol_ratio": ind["vol_ratio"],
            "market_regime": market["regime"], "divergence": div_lbl,
            "supertrend": st_lbl, "vwap": vwap,
            "fib_levels": {"23.6%": ind.get("fib_236"), "38.2%": ind.get("fib_382"),
                           "50%": ind.get("fib_500"), "61.8%": ind.get("fib_618")},
            "expected_hold": "5-8 Days" # <--- ADD THIS LINE
            })
            continue

        cmp, rsi = ind["cmp"], ind["rsi"]
        ema20, ema50, atr = ind["ema20"], ind["ema50"], ind["atr"]
        support, resistance = ind["support"], ind["resistance"]
        trend = ind["trend"]
        macd_bull, macd_bear = ind["macd_bullish"], ind["macd_bearish"]
        bb_pos, bb_lower = ind["bb_pos"], ind["bb_lower"]
        rsi_div, macd_div = ind["rsi_divergence"], ind["macd_divergence"]
        st_bullish, st_val = ind.get("supertrend_bullish"), ind.get("supertrend")
        vwap = ind.get("vwap")
        patterns = ind.get("patterns", [])
        candles = ind.get("candlesticks", [])

        pct = round((cmp - buy_at) / buy_at * 100, 2)
        near52h = cmp >= ind["high52"] * 0.97
        near52l = cmp <= ind["low52"] * 1.03
        nifty_chg = market.get("indices", {}).get("Nifty 50", {}).get("chg_pct", 0)
        stock_rs_strong = pct > nifty_chg 

        initial_stop = round(buy_at - 2 * atr, 2)
        if st_bullish and st_val:
            trail_stop = round(max(buy_at, st_val), 2) if pct > 0 else round(st_val, 2)
        else:
            trail_stop = round(max(initial_stop, cmp - 2.5 * atr), 2)

        sl_sup = round(support - atr*0.5, 2)

        # SELL
        sell = []
        if rsi and rsi >= 75: sell.append(f"RSI overbought ({rsi})")
        if rsi and rsi >= 70 and near52h: sell.append("Near 52w high")
        if cmp < trail_stop: sell.append(f"Below Trail Stop (₹{trail_stop})")
        elif not st_bullish: sell.append("Supertrend Bearish")
            
        if ema50 and cmp < ema50 and pct < -5: sell.append("Below EMA50 (-5% loss)")
        if macd_bear: sell.append("MACD Bearish Cross")
        if rsi_div["bearish_div"]: sell.append("RSI Bear Div")
        if macd_div["bearish_div"]: sell.append("MACD Bear Div")
        
        if "📈 Double Top" in patterns: sell.append("Double Top Rejection")
        if "🏔️ Head & Shoulders (Top)" in patterns: sell.append("H&S Bearish Reversal")
        if "🟥 Bearish Engulfing" in candles and rsi > 65: sell.append("Bearish Institutional Distribution Candle")
        
        if ind["vol_ratio"] > 2.5 and rsi and rsi > 65: sell.append("Volume spike at resistance")
        if trend in ("Downtrend", "Strong Downtrend") and pct < -8: sell.append(f"{trend} breakdown")
        if is_bear and pct < -5: sell.append("Bear Market override")

        # AVERAGE / BUY 
        avg = []
        can_avg = not (trend == "Strong Downtrend" and not ("📉 Double Bottom" in patterns or "🛤️ Inverse H&S (Bottom)" in patterns))
        if can_avg:
            if rsi and rsi <= 40 and pct < -5:
                if "🔨 Bullish Hammer" in candles or "🟩 Bullish Engulfing" in candles:
                    avg.append(f"Oversold Bounce Confirmed by {candles[0]}")
            if "📉 Double Bottom" in patterns and stock_rs_strong: avg.append("Double Bottom + Relative Strength")
            if "🛤️ Inverse H&S (Bottom)" in patterns: avg.append("Inverse H&S Reversal")
            if trend in ("Uptrend", "Strong Uptrend") and cmp <= ema20 * 1.015 and pct < -3:
                if "🔨 Bullish Hammer" in candles: avg.append("EMA20 Pullback + Hammer Confirmation")
            if "🚀 Vol Breakout" in patterns and pct < 0: avg.append("Reversal Vol Breakout")
            if "🚩 Bull Flag Breakout" in patterns and pct < 0: avg.append("Bull Flag Breakout")

        # HOLD
        hold = []
        if "🚀 Vol Breakout" in patterns and pct > 0: hold.append("Bullish Breakout (Hold tight)")
        if "🚩 Bull Flag Breakout" in patterns and pct > 0: hold.append("Bull Flag Continuation")
        if rsi and 45 <= rsi <= 65 and cmp > ema20 and stock_rs_strong: hold.append(f"RSI neutral, Above EMA20, Beating Index")
        elif pct > 0 and not sell: hold.append(f"In profit {pct:+.1f}%")
        elif st_bullish and trend in ("Uptrend", "Strong Uptrend") and not sell: hold.append("Trend & Supertrend Bullish")

        # Determine action
        if sell:
            action, reason_base, strength = "🔴 SELL", " | ".join(sell), min(len(sell)*25+15, 95)
            target, stop_loss = cmp, round(min(trail_stop, sl_sup), 2)
            avg_price = new_avg = new_sl = None
        elif avg:
            action, reason_base, strength = "🟡 AVERAGE", " | ".join(avg), min(len(avg)*22+18, 88)
            avg_price = cmp
            new_avg = round((buy_at*qty + cmp*qty)/(qty+qty), 2)
            new_sl = round(new_avg - 2*atr, 2)
            fib382 = ind.get("fib_382", resistance)
            target = round(max(resistance, fib382), 2)
            stop_loss = round(sl_sup, 2)
        elif hold:
            action, reason_base, strength = "🟢 HOLD", hold[0], 55
            target = round(resistance, 2)
            stop_loss = round(max(trail_stop, sl_sup), 2)
            avg_price = new_avg = new_sl = None
        else:
            action = "⚪ WATCH"
            reason_base = f"CMP ₹{cmp} | RSI {rsi:.2f} | {pct:+.1f}%"
            strength, target, stop_loss = 30, round(resistance, 2), round(sl_sup, 2)
            avg_price = new_avg = new_sl = None

        risk = cmp - stop_loss if cmp and stop_loss else 0
        reward = target - cmp if target and cmp else 0
        rr = round(reward/risk, 2) if risk > 0 else None

        div_parts = []
        if rsi_div["bullish_div"]: div_parts.append("RSI Bull")
        if rsi_div["bearish_div"]: div_parts.append("RSI Bear")
        if macd_div["bullish_div"]: div_parts.append("MACD Bull")
        if macd_div["bearish_div"]: div_parts.append("MACD Bear")
        div_lbl = ", ".join(div_parts) if div_parts else "None"

        macd_lbl = "Bullish ↗" if macd_bull else ("Bearish ↘" if macd_bear else "Neutral →")
        bb_lbl = "Lower" if bb_pos < 0.2 else ("Upper" if bb_pos > 0.8 else "Mid")
        st_lbl = f"Bullish ₹{st_val}" if st_bullish else (f"Bearish ₹{st_val}" if st_val else "—")

        all_pats = patterns + candles
        pat_str = " | ".join(all_pats) if all_pats else ""
        final_reason = f"[{pat_str}] {reason_base}" if pat_str else reason_base

        signals.append({
            "id": tid, "stock": symbol, "sector": get_sector(symbol),
            "action": action, "reason": final_reason, "strength": strength,
            "cmp": cmp, "rsi": rsi, "ema20": ema20, "ema50": ema50, "atr": atr,
            "pct_from_buy": pct, "buy_at": buy_at, "quantity": qty,
            "target": target, "stop_loss": stop_loss,
            "avg_price": avg_price, "new_avg": new_avg, "new_sl": new_sl,
            "macd_signal": macd_lbl, "bb_position": bb_lbl,
            "trend": trend, "support": support, "resistance": resistance,
            "risk_reward": rr, "vol_ratio": ind["vol_ratio"],
            "market_regime": market["regime"], "divergence": div_lbl,
            "supertrend": st_lbl, "vwap": vwap,
            "fib_levels": {"23.6%": ind.get("fib_236"), "38.2%": ind.get("fib_382"),
                           "50%": ind.get("fib_500"), "61.8%": ind.get("fib_618")},
        })
    return signals
    
# ─── UPGRADED SECTOR ROTATION ────────────────────────────────────────────────
# Adds: relative strength vs Nifty, 4-quadrant RRG, trend duration score
def sector_rotation(trades_df=None):
    rows = []
    idx_symbols = list(SECTOR_INDICES.values()) + ["^NSEI"]
    bulk_data   = _bulk_fetch_history(idx_symbols, period="6mo")

    # Benchmark: Nifty 50 returns
    nifty_df    = bulk_data.get("^NSEI")
    nifty_ret1m = 0.0
    nifty_ret3m = 0.0
    if nifty_df is not None and len(nifty_df) >= 20:
        nifty_ret1m = (float(nifty_df["Close"].iloc[-1]) / float(nifty_df["Close"].iloc[-21]) - 1) * 100
    if nifty_df is not None and len(nifty_df) >= 60:
        nifty_ret3m = (float(nifty_df["Close"].iloc[-1]) / float(nifty_df["Close"].iloc[-61]) - 1) * 100

    for sector, idx_sym in SECTOR_INDICES.items():
        try:
            df_idx = bulk_data.get(idx_sym)
            if df_idx is None or len(df_idx) < 21:
                continue

            close   = df_idx["Close"]
            cmp_now = float(close.iloc[-1])
            rsi     = compute_rsi(close)
            ema20   = float(close.ewm(span=20).mean().iloc[-1])
            ema50   = float(close.ewm(span=50).mean().iloc[-1]) if len(close) >= 50 else ema20

            # Returns at multiple timeframes
            ret_1w  = (cmp_now / float(close.iloc[-6])  - 1) * 100 if len(close) >= 6  else 0.0
            ret_1m  = (cmp_now / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else 0.0
            ret_3m  = (cmp_now / float(close.iloc[-61]) - 1) * 100 if len(close) >= 61 else 0.0

            # ── Relative Strength vs Nifty (key rotation signal) ─────────────
            rs_1m   = round(ret_1m - nifty_ret1m, 2)
            rs_3m   = round(ret_3m - nifty_ret3m, 2)

            # ── RRG Quadrant (Relative Rotation Graph) ────────────────────────
            # RS-Ratio  > 100 = outperforming; RS-Momentum > 100 = RS accelerating
            rs_ratio    = 100 + rs_3m / 10        # simplified RRG x-axis
            rs_momentum = 100 + rs_1m / 5         # simplified RRG y-axis

            if   rs_ratio > 100 and rs_momentum > 100:
                rrg_quadrant = "🔥 Leading"        # outperforming and accelerating
            elif rs_ratio > 100 and rs_momentum <= 100:
                rrg_quadrant = "📉 Weakening"      # outperforming but losing steam
            elif rs_ratio <= 100 and rs_momentum > 100:
                rrg_quadrant = "🔄 Improving"      # underperforming but turning up
            else:
                rrg_quadrant = "❄️ Lagging"        # underperforming and decelerating

            # ── Trend duration (how long above EMA50) ────────────────────────
            above_ema50   = (close > close.ewm(span=50).mean()).astype(int)
            streak        = 0
            for val in reversed(above_ema50.values):
                if val == 1:
                    streak += 1
                else:
                    break
            trend_duration_days = streak

            # ── Composite momentum score ──────────────────────────────────────
            rsi_score      = (rsi / 100) if rsi else 0.5
            rs_score       = max(-1.0, min(1.0, rs_1m / 10))  # clip to [-1, 1]
            trend_score    = min(1.0, trend_duration_days / 60)
            macd_score     = 1.0 if cmp_now > ema20 > ema50 else (0.5 if cmp_now > ema20 else 0.0)

            momentum_score = round(
                rsi_score  * 0.20 +
                rs_score   * 0.40 +   # relative strength is the dominant factor
                trend_score * 0.20 +
                macd_score  * 0.20,
                3
            )

            rows.append({
                "sector":            sector,
                "stocks":            ", ".join(SECTOR_STOCKS.get(sector, [])[:4]) + "...",
                "cmp":               round(cmp_now, 2),
                "rsi":               rsi,
                "ret_1w":            round(ret_1w, 2),
                "ret_1m":            round(ret_1m, 2),
                "ret_3m":            round(ret_3m, 2),
                "rs_vs_nifty_1m":    rs_1m,
                "rs_vs_nifty_3m":    rs_3m,
                "rrg_quadrant":      rrg_quadrant,
                "trend_days":        trend_duration_days,
                "momentum_score":    momentum_score,
                "macd_bullish":      cmp_now > ema20,
                "count":             len(SECTOR_STOCKS.get(sector, [])),
            })

        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("momentum_score", ascending=False)
    df["rank"]          = range(1, len(df) + 1)
    df["avg_rsi"]       = df["rsi"]
    df["avg_pct"]       = df["ret_1m"]
    df["bullish_count"] = df["macd_bullish"].astype(int)
    df["index_chg"]     = df["ret_1m"]

    return df[[
        "rank", "sector", "stocks", "count",
        "avg_rsi", "avg_pct", "rs_vs_nifty_1m", "rs_vs_nifty_3m",
        "rrg_quadrant", "trend_days", "momentum_score",
        "bullish_count", "index_chg"
    ]]

# ─── Sector Outlook ───────────────────────────────────────────────────────────
def predict_sector_outlook(sector_df):
    if sector_df.empty: return pd.DataFrame()
    preds = []
    for _, r in sector_df.iterrows():
        score = r["momentum_score"]*0.4 + (r.get("index_chg",0)/10 if r.get("index_chg") else 0)*0.3 + (r["bullish_count"]/max(r["count"],1))*0.3
        if score > 0.5: outlook, conf = "🔥 Strong Bullish", 85
        elif score > 0.3: outlook, conf = "📈 Bullish", 70
        elif score > 0.1: outlook, conf = "➡️ Neutral-Bullish", 55
        elif score > -0.1: outlook, conf = "➡️ Neutral", 50
        elif score > -0.3: outlook, conf = "📉 Weak", 30
        else: outlook, conf = "🔻 Bearish", 20

        # --- REWARD CAPITAL ROTATION INTO POWER ZONES ---
        if r["avg_rsi"] and r["avg_rsi"] > 65: 
            outlook = "🚀 Power Zone"
            conf = min(conf + 15, 95)
        elif r["avg_rsi"] and r["avg_rsi"] < 45: 
            outlook = "🩸 Bleeding — Avoid"
            conf = max(conf - 20, 20)
        preds.append({"sector": r["sector"], "outlook": outlook, "confidence": conf,
                      "momentum": r["momentum_score"], "avg_rsi": r["avg_rsi"],
                      "avg_pct": r["avg_pct"], "index_chg": r.get("index_chg")})
    return pd.DataFrame(preds).sort_values("confidence", ascending=False)

# ─── Sector Stock Discovery ───────────────────────────────────────────────────
def find_sector_picks(selected_sectors=None, max_per_sector=3):
    picks, sectors = [], selected_sectors or list(SECTOR_STOCKS.keys())
    
    all_symbols = []
    for sector in sectors:
        all_symbols.extend(SECTOR_STOCKS.get(sector, []))
        
    bulk_data = _bulk_fetch_history(all_symbols, period="1y")

    for sector in sectors:
        spicks = []
        for symbol in SECTOR_STOCKS.get(sector, []):
            df = bulk_data.get(symbol)
            ind = compute_indicators(symbol, period="1y", prefetched_df=df)
            
            if ind is None: continue
            cmp, rsi = ind["cmp"], ind["rsi"]
            score, reasons = 0, []
            
            if rsi and 35 <= rsi <= 55: score += 15; reasons.append(f"RSI buy zone ({rsi})")
            if ind["trend"] in ("Uptrend","Strong Uptrend"): score += 20; reasons.append(ind["trend"])
            elif ind["trend"] == "Recovery": score += 12; reasons.append("Recovery")
            if ind["macd_bullish"]: score += 15; reasons.append("MACD bullish")
            if ind.get("supertrend_bullish"): score += 10; reasons.append("Supertrend bullish")
            if ind["bb_pos"] < 0.3: score += 8; reasons.append("Lower BB bounce")
            if ind["vol_ratio"] > 1.3: score += 8; reasons.append(f"Vol surge ({ind['vol_ratio']:.1f}x)")
            
            if score < 45: continue
            
            # --- UPDATED FOR 10% TARGET IN 5 DAYS ---
            # Stop Loss tightened to 1.25 ATR for quick 1-week invalidation
            sl = round(cmp - (1.25 * ind["atr"]), 2)
            
            # Target set to a baseline 10% (1.10) or immediate resistance
            tgt = round(max(cmp * 1.10, cmp + (2.5 * ind["atr"])), 2) 
            
            risk, reward = cmp - sl, tgt - cmp
            rr = round(reward / risk, 2) if risk > 0 else None
            
            # Accept trades with a solid 1.5 R:R to allow for 10% base hits
            if rr and rr < 1.5: continue
            
            if rr and rr < 1.5: continue
            
            patterns = ind.get("patterns", [])
            candles = ind.get("candlesticks", [])
            all_pats = patterns + candles
            pat_str = " | ".join(all_pats) if all_pats else ""
            final_reason = f"[{pat_str}] " + " | ".join(reasons[:4]) if pat_str else " | ".join(reasons[:4])

            spicks.append({"stock": symbol, "sector": sector, "cmp": cmp, "entry": round(cmp,2),
                           "target": tgt, "stop_loss": sl, "risk_reward": rr, "score": score,
                           "rsi": rsi, "trend": ind["trend"], "reason": final_reason,
                           "atr": ind["atr"], "support": ind["support"], "resistance": ind["resistance"], "expected_hold": "5-8 Days" # <--- ADD THIS LINE})
                           
        spicks.sort(key=lambda x: x["score"], reverse=True)
        picks.extend(spicks[:max_per_sector])
        
    picks.sort(key=lambda x: x["score"], reverse=True)
    return picks

# ─── Telegram ─────────────────────────────────────────────────────────────────
def send_telegram(bot_token, chat_id, message):
    try:
        resp = requests.post(f"https://api.telegram.org/bot{bot_token}/sendMessage",
                             json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=10)
        return resp.ok
    except: return False

def build_telegram_message(signals, sector_df, picks=None):
    now = datetime.now().strftime("%d %b %Y %H:%M")
    lines = [f"<b>📈 Swing Dashboard</b>  <i>{now}</i>\n"]
    market = get_market_regime()
    lines.append(f"🌐 <b>Market:</b> {market['regime']} | Nifty ₹{market.get('nifty_close','—')} | RSI {market.get('nifty_rsi','—')}")
    for s in [s for s in signals if "SELL" in s["action"]]:
        lines.append(f"🔴 <b>{s['stock']}</b> ₹{s['cmp']} | {s['reason']}")
    for s in [s for s in signals if "AVERAGE" in s["action"]]:
        lines.append(f"🟡 <b>{s['stock']}</b> ₹{s['cmp']} | Avg ₹{s.get('avg_price')} | New SL ₹{s.get('new_sl')}")
    for s in [s for s in signals if "HOLD" in s["action"]]:
        lines.append(f"🟢 <b>{s['stock']}</b> ₹{s['cmp']} | {s['reason']}")
    if not sector_df.empty:
        lines.append("\n🔄 <b>SECTORS</b>")
        for _, r in sector_df.head(5).iterrows():
            e = "🥇" if r["rank"]==1 else "🥈" if r["rank"]==2 else "📊"
            lines.append(f"  {e} <b>{r['sector']}</b> Score {r['momentum_score']:.2f}")
    if picks:
        lines.append("\n🎯 <b>BUYS</b>")
        for p in picks[:8]: lines.append(f"  • <b>{p['stock']}</b> ₹{p['cmp']} | Tgt ₹{p['target']} | SL ₹{p['stop_loss']} | R:R {p['risk_reward']}")
    lines.append("\n<i>Indicative only.</i>")
    return "\n".join(lines)

# ─── Master Universe Scanner ──────────────────────────────────────────────────
def generate_market_scanner():
    """Scans the entire F&O universe and generates independent setup signals."""
    all_symbols = []
    for sector, stocks in SECTOR_STOCKS.items():
        all_symbols.extend(stocks)
        
    bulk_data = _bulk_fetch_history(all_symbols, period="6mo") 
    
    results = []
    for symbol in all_symbols:
        sector = get_sector(symbol)
        df = bulk_data.get(symbol)
        ind = compute_indicators(symbol, period="6mo", prefetched_df=df)
        
        if not ind: 
            continue
            
        cmp, rsi, trend = ind["cmp"], ind["rsi"], ind["trend"]
        patterns = ind.get("patterns", [])
        candles = ind.get("candlesticks", [])
        
        # --- CONFLUENCE SCORING ENGINE FOR 1-WEEK EXPLOSIVE ENTRIES ---
        score = 0
        
        if trend in ("Uptrend", "Strong Uptrend"): score += 3
        if ind.get("supertrend_bullish"): score += 2
        if ind.get("macd_bullish"): score += 2
        
        # BUY HIGH, SELL HIGHER: Reward breakouts entering overbought
        if rsi and 60 <= rsi <= 75: score += 3 
        
        if "🚀 Vol Breakout" in patterns: score += 5
        if "🚩 Bull Flag Breakout" in patterns: score += 4
        if "☕ Cup & Handle Breakout" in patterns: score += 4
        if "🟩 Bullish Engulfing" in candles: score += 2
        
        if "📈 Double Top" in patterns or "🏔️ Head & Shoulders (Top)" in patterns or "🟥 Bearish Engulfing" in candles:
            score -= 5
        if trend in ("Downtrend", "Strong Downtrend"):
            score -= 4
            
        # Signal Generation
        if score >= 8: signal = "🔥 STRONG BUY"
        elif score >= 5: signal = "🟢 BUY SETUP"
        elif score >= 2: signal = "🟡 ACCUMULATE"
        elif score <= 0: signal = "🔴 AVOID"
        else: signal = "⚪ NEUTRAL"
        
        all_pats = patterns + candles
        pat_str = " | ".join(all_pats) if all_pats else "—"
        
        # --- CALCULATE STRICT 1-WEEK RISK METRICS (10% TARGET, 1.25 ATR SL) ---
        atr = ind["atr"]
        sup, res = ind["support"], ind["resistance"]
        
        sl = round(cmp - (1.25 * atr), 2)
        tgt = round(max(cmp * 1.10, cmp + (2.5 * atr)), 2)
        
        results.append({
            "Generated": datetime.now().strftime("%d %b %H:%M"),
            "Sector": sector,
            "Stock": symbol,
            "CMP": float(cmp),          
            "Entry": float(cmp),
            "Target": float(tgt),
            "SL": float(sl),
            "Support": float(sup),
            "Resist": float(res),
            "Signal": signal,
            "Score": score,
            "RSI": round(float(rsi), 2) if rsi else 0.0, 
            "Trend": trend,
            "Patterns": pat_str,
            "expected_hold": "5-8 Days" # <--- ADD THIS LINE
        })
        
    return pd.DataFrame(results).sort_values(by=["Sector", "Score"], ascending=[True, False])

def fetch_portfolio_news(open_trades_df):
    """
    Acts as a mini-agent to scrape the latest headlines for active holdings.
    """
    if open_trades_df.empty:
        return []
        
    news_alerts = []
    unique_symbols = open_trades_df["stock"].unique().tolist()
    
    for sym in unique_symbols:
        try:
            # 1. Strip any existing suffixes to prevent SNOWMAN.NS.NS
            clean_sym = str(sym).upper().strip()
            for sfx in [".NS", ".BO", ".NSE", ".BSE"]:
                if clean_sym.endswith(sfx):
                    clean_sym = clean_sym[:-len(sfx)]
            
            # 2. Fetch using the safe base symbol
            t = yf.Ticker(f"{clean_sym}.NS")
            news = t.news
            
            if news and len(news) > 0:
                top_story = news[0]
                title = top_story.get("title", "")
                link = top_story.get("link", "")
                
                # 3. Only add if it successfully retrieves a headline
                if title:
                    # Format safely as HTML if link exists
                    if link:
                        news_alerts.append(f"📰 <b>{clean_sym}</b>: <a href='{link}' style='color:var(--accent); text-decoration:none;'>{title}</a>")
                    else:
                        news_alerts.append(f"📰 <b>{clean_sym}</b>: {title}")
        except Exception:
            continue
            
    return news_alerts
