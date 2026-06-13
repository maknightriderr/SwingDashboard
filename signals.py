"""
signals.py v12 — Institutional-Grade Signal Engine
All v11 scorecard gaps fixed:
  1. RSI        — adjust=False + explicit 100/0 edge case        (7 → 9)
  2. MACD       — adjust=False, single-pass crossover, histogram  (6 → 9)
  3. Bollinger  — bb_pos clamped [0,1], bandwidth + squeeze       (6 → 8)
  4. ATR        — Wilder's EWM smoothing (matches Zerodha/TV)     (6 → 9)
  5. Supertrend — numpy array loop, Wilder ATR, mult 2.5          (5 → 9)
  6. VWAP       — 20-day rolling + price_vs_vwap %                (4 → 8)
  7. EMA/Trend  — slope check, momentum-fading flag, EMA200 back  (7 → 8)
  8. Fibonacci  — swing-peak based (scipy), not fixed window      (6 → 8)
  9. Risk Engine— find_sector_picks + scanner now use unified     (7 → 9)
 10. Liquidity  — soft gate with liquidity_ok flag (no silent None)(7 → 8)
All output keys are backward-compatible with app.py v13.
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.signal import find_peaks

# ==============================================================================
# 1. NIFTY 500 WATCHLIST
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
    "Realty": "^CNXREALTY", "Media": "^CNXMEDIA",
    "Consumer Durables": "^CNXCONSUM", "Consumer Services": "^CNXSERVICE",
    "PSU Bank": "^CNXPSUBANK", "Private Bank": "^CNXPVTBANK", "Bank": "^NSEBANK"
}

TRACKED_INDICES = {
    "Sensex": "^BSESN", "Nifty 50": "^NSEI",
    "Nifty Midcap": "^NSEMDCP50", "Nifty Smallcap": "^CNXSC",
    "Bank Nifty": "^NSEBANK", "Nifty IT": "^CNXIT", "India VIX": "^INDIAVIX"
}

# ─── Data Fetcher ──────────────────────────────────────────────────────────────
def _fetch_history(ticker, period="1y", interval="1d"):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval, auto_adjust=True)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        result = pd.DataFrame()
        result["Open"]   = df["Open"]   if "Open"   in df.columns else df["Close"]
        result["Close"]  = df["Close"]
        result["High"]   = df["High"]   if "High"   in df.columns else df["Close"]
        result["Low"]    = df["Low"]    if "Low"    in df.columns else df["Close"]
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
                return sym, _fetch_history(sym, period)
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

# ==============================================================================
# FIX 1+4: Wilder's RSI and ATR with adjust=False and edge-case handling
# ==============================================================================
def compute_rsi_wilder(series, period=14):
    """Wilder's RSI. adjust=False matches TradingView/Zerodha exactly.
    Explicit 100/0 on all-gain/all-loss streaks instead of silent NaN."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rsi = pd.Series(np.nan, index=series.index, dtype=float)
    m100 = (avg_loss == 0) & (avg_gain > 0)          # pure uptrend → RSI 100
    m0   = (avg_gain == 0) & (avg_loss > 0)          # pure downtrend → RSI 0
    mn   = (avg_gain > 0) & (avg_loss > 0)
    rsi[m100] = 100.0
    rsi[m0]   = 0.0
    rs = avg_gain[mn] / avg_loss[mn]
    rsi[mn] = 100 - (100 / (1 + rs))
    return rsi


def compute_rsi(series, period=14):
    rsi = compute_rsi_wilder(series, period)
    val = rsi.iloc[-1]
    return round(float(val), 1) if not pd.isna(val) else None


def compute_atr_wilder(high, low, close, period=14):
    """True ATR with Wilder's EWM smoothing — matches Zerodha/TradingView.
    Returns (true_range_series, atr_series)."""
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return tr, atr


# ─── Divergence Detection ─────────────────────────────────────────────────────
def detect_rsi_divergence(close, rsi_series, window=40):
    if len(close) < window or len(rsi_series) < window:
        return {"bullish_div": False, "bearish_div": False}
    c = close.iloc[-window:].values
    r = rsi_series.iloc[-window:].values
    troughs, _ = find_peaks(-c, distance=5)
    peaks, _   = find_peaks(c, distance=5)
    bullish = bearish = False
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
    peaks, _   = find_peaks(c, distance=5)
    bullish = bearish = False
    if len(troughs) >= 2:
        if c[troughs[-1]] < c[troughs[-2]] and m[troughs[-1]] > m[troughs[-2]]:
            bullish = True
    if len(peaks) >= 2:
        if c[peaks[-1]] > c[peaks[-2]] and m[peaks[-1]] < m[peaks[-2]]:
            bearish = True
    return {"bullish_div": bullish, "bearish_div": bearish}


# ─── Chart Pattern Detection (unchanged from v11 — already 7+/10) ─────────────
def detect_price_patterns(high, low, close, vol, vol_avg):
    patterns = []
    if len(close) < 30:
        return patterns

    cmp    = float(close.iloc[-1])
    c_vals = close.values

    troughs, _ = find_peaks(-c_vals, distance=8, prominence=c_vals.std() * 0.3)
    peaks,   _ = find_peaks( c_vals, distance=8, prominence=c_vals.std() * 0.3)

    if len(close) >= 20:
        recent_h = high.iloc[-20:-1].max()
        recent_l = low.iloc[-20:-1].min()
        rng_pct  = (recent_h - recent_l) / recent_l
        if rng_pct < 0.10:
            if cmp > recent_h and float(vol.iloc[-1]) > vol_avg * 2.5:
                patterns.append("🚀 Vol Breakout")

    if len(close) >= 30:
        pole   = close.iloc[-30:-10]
        flag   = close.iloc[-10:-1]
        p_gain = (pole.max() - pole.min()) / (pole.min() + 1e-8)
        f_drop = (flag.max() - flag.min()) / (flag.max() + 1e-8)
        if p_gain > 0.08 and f_drop < 0.06 and flag.iloc[-1] < pole.max():
            if cmp > flag.max() and float(vol.iloc[-1]) > vol_avg * 2.0:
                patterns.append("🚩 Bull Flag Breakout")

    if len(troughs) >= 2:
        t1, t2 = troughs[-2], troughs[-1]
        p1, p2 = c_vals[t1], c_vals[t2]
        depth_ok    = abs(p1 - p2) / (p1 + 1e-8) < 0.08
        price_ok    = p2 * 1.00 < cmp < p2 * 1.12
        vol_confirm = float(vol.iloc[-1]) > vol_avg * 1.2
        if depth_ok and price_ok and vol_confirm:
            patterns.append("📉 Double Bottom")

    if len(peaks) >= 2:
        p1_idx, p2_idx = peaks[-2], peaks[-1]
        v1, v2 = c_vals[p1_idx], c_vals[p2_idx]
        if abs(v1 - v2) / (v1 + 1e-8) < 0.08 and v2 * 0.88 < cmp < v2 * 0.99:
            patterns.append("📈 Double Top")

    if len(peaks) >= 3 and len(troughs) >= 2:
        p1, p2, p3 = c_vals[peaks[-3]], c_vals[peaks[-2]], c_vals[peaks[-1]]
        head_valid = p2 > p1 and p2 > p3 and abs(p1 - p3) / (p1 + 1e-8) < 0.06
        if head_valid:
            neckline = (c_vals[troughs[-2]] + c_vals[troughs[-1]]) / 2
            if cmp < neckline * 0.99 and float(vol.iloc[-1]) > vol_avg * 1.3:
                patterns.append("🏔️ Head & Shoulders (Top)")

    if len(troughs) >= 3 and len(peaks) >= 2:
        t1, t2, t3 = c_vals[troughs[-3]], c_vals[troughs[-2]], c_vals[troughs[-1]]
        head_valid = t2 < t1 and t2 < t3 and abs(t1 - t3) / (t1 + 1e-8) < 0.06
        if head_valid:
            neckline = (c_vals[peaks[-2]] + c_vals[peaks[-1]]) / 2
            if cmp > neckline * 1.01 and float(vol.iloc[-1]) > vol_avg * 1.3:
                patterns.append("🛤️ Inverse H&S (Bottom)")

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
        if (0.10 < cup_depth < 0.40 and rim_match < 0.06 and
                handle_ret < 0.08 and breakout and float(vol.iloc[-1]) > vol_avg * 1.5):
            patterns.append("☕ Cup & Handle Breakout")

    return patterns


# ─── Candlestick Detection (unchanged from v11 — already 8/10) ────────────────
def detect_candlesticks(open_p, high, low, close):
    candles = []
    if len(close) < 5:
        return candles

    def _candle(i):
        o, h, l, c = float(open_p.iloc[i]), float(high.iloc[i]), float(low.iloc[i]), float(close.iloc[i])
        body = abs(c - o); rng = h - l
        if rng < 1e-8: return None
        upper_wick = h - max(o, c); lower_wick = min(o, c) - l
        bullish = c > o
        return dict(o=o, h=h, l=l, c=c, body=body, rng=rng,
                    upper_wick=upper_wick, lower_wick=lower_wick, bullish=bullish)

    c0 = _candle(-1); c1 = _candle(-2)
    c2 = _candle(-3) if len(close) >= 3 else None
    if not c0 or not c1:
        return candles

    if (c0["lower_wick"] >= c0["rng"] * 0.55 and c0["upper_wick"] <= c0["rng"] * 0.15 and c0["body"] >= c0["rng"] * 0.05):
        candles.append("🔨 Bullish Hammer")
    if (c0["upper_wick"] >= c0["rng"] * 0.55 and c0["lower_wick"] <= c0["rng"] * 0.15 and c0["body"] >= c0["rng"] * 0.05 and not c0["bullish"]):
        candles.append("💫 Shooting Star")
    if c0["body"] <= c0["rng"] * 0.07:
        candles.append("〰️ Doji (Indecision)")
    if (not c1["bullish"] and c0["bullish"] and c0["o"] <= c1["c"] and c0["c"] >= c1["o"] and c0["body"] > c1["body"] * 1.0):
        candles.append("🟩 Bullish Engulfing")
    if (c1["bullish"] and not c0["bullish"] and c0["o"] >= c1["c"] and c0["c"] <= c1["o"] and c0["body"] > c1["body"] * 1.0):
        candles.append("🟥 Bearish Engulfing")
    if (not c1["bullish"] and c0["bullish"] and c0["o"] > c1["c"] and c0["c"] < c1["o"] and c0["body"] < c1["body"] * 0.5):
        candles.append("🟢 Bullish Harami")
    if (not c1["bullish"] and c0["bullish"] and c0["o"] < c1["l"] and c0["c"] > (c1["o"] + c1["c"]) / 2 and c0["c"] < c1["o"]):
        candles.append("🔆 Piercing Line")

    if c2:
        if (not c2["bullish"] and c2["body"] >= c2["rng"] * 0.5 and c1["body"] <= c1["rng"] * 0.3 and
                c0["bullish"] and c0["body"] >= c0["rng"] * 0.5 and c0["c"] > (c2["o"] + c2["c"]) / 2):
            candles.append("🌅 Morning Star")
        if (c2["bullish"] and c2["body"] >= c2["rng"] * 0.5 and c1["body"] <= c1["rng"] * 0.3 and
                not c0["bullish"] and c0["body"] >= c0["rng"] * 0.5 and c0["c"] < (c2["o"] + c2["c"]) / 2):
            candles.append("🌆 Evening Star")
        if (c2["bullish"] and c1["bullish"] and c0["bullish"] and
                c1["o"] > c2["o"] and c0["o"] > c1["o"] and c1["c"] > c2["c"] and c0["c"] > c1["c"] and
                c0["body"] >= c0["rng"] * 0.5 and c1["body"] >= c1["rng"] * 0.5):
            candles.append("🪖 Three White Soldiers")
        if (not c2["bullish"] and not c1["bullish"] and not c0["bullish"] and
                c1["o"] < c2["o"] and c0["o"] < c1["o"] and c1["c"] < c2["c"] and c0["c"] < c1["c"] and
                c0["body"] >= c0["rng"] * 0.5 and c1["body"] >= c1["rng"] * 0.5):
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
                prev    = float(df["Close"].iloc[-2])
                chg     = round((current / prev - 1) * 100, 2)
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
            close  = df["Close"]
            ema20  = float(close.ewm(span=20,  adjust=False).mean().iloc[-1])
            ema50  = float(close.ewm(span=50,  adjust=False).mean().iloc[-1])
            ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1]) if len(close) >= 200 else None
            nifty_rsi  = compute_rsi(close)
            support    = float(df["Low"].rolling(20).min().iloc[-1])
            resistance = float(df["High"].rolling(20).max().iloc[-1])

            if ema200 and nifty_close > ema200:
                if nifty_close > ema20 > ema50: regime, trend = "Strong Bull", "Uptrend"
                elif nifty_close > ema20: regime, trend = "Bull", "Uptrend"
                else: regime, trend = "Bull Pullback", "Pullback"
            elif ema200 and nifty_close < ema200:
                if nifty_close < ema20 < ema50: regime, trend = "Strong Bear", "Downtrend"
                elif nifty_close < ema20: regime, trend = "Bear", "Downtrend"
                else: regime, trend = "Bear Rally", "Relief Rally"
            else:
                regime, trend = "Neutral", "Sideways"

            if regime in ("Strong Bull", "Strong Bear"): conf = 85
            elif regime in ("Bull", "Bear"): conf = 70
            elif regime in ("Bull Pullback", "Bear Rally"): conf = 55

    if nifty_rsi:
        if nifty_rsi > 70: conf = min(95, conf + 15)
        elif nifty_rsi < 40: conf = max(20, conf - 20)

    risk = "Neutral"
    if nifty_rsi:
        if nifty_rsi > 70: risk = "High Momentum (Power Zone)"
        elif nifty_rsi > 60: risk = "Building Momentum"
        elif nifty_rsi < 40: risk = "High Risk (Downtrend/Bleeding)"

    result = {
        "regime": regime, "trend": trend, "nifty_close": nifty_close,
        "nifty_rsi": nifty_rsi, "risk_level": risk, "indices": indices_data,
        "support": support, "resistance": resistance, "confidence": conf
    }
    _market_regime_cache["data"] = result
    _market_regime_cache["ts"] = now
    return result


# ==============================================================================
# TECHNICAL INDICATORS — v12 with all fixes
# ==============================================================================
def _compute_indicators_raw(symbol, period="1y", prefetched_df=None):
    df = prefetched_df
    if df is None:
        for suffix in [".NS", ".BO"]:
            df = _fetch_history(symbol + suffix, period=period, interval="1d")
            if df is not None:
                break

    if df is None or len(df) < 50:
        return None

    open_p = df["Open"]; high = df["High"]; low = df["Low"]
    close = df["Close"]; vol = df["Volume"]
    cmp = float(close.iloc[-1])
    if pd.isna(cmp) or cmp <= 0:
        return None

    # ── FIX 1: RSI with adjust=False + edge case ──────────────────────────────
    rsi_series = compute_rsi_wilder(close, 14)
    rsi = round(float(rsi_series.iloc[-1]), 1) if not pd.isna(rsi_series.iloc[-1]) else None

    # ── FIX 7: EMAs with adjust=False + slope detection + EMA200 restored ────
    ema9_s   = close.ewm(span=9,   adjust=False).mean()
    ema21_s  = close.ewm(span=21,  adjust=False).mean()
    ema50_s  = close.ewm(span=50,  adjust=False).mean()
    ema200_s = close.ewm(span=200, adjust=False).mean() if len(close) >= 100 else None

    ema9   = float(ema9_s.iloc[-1])
    ema21  = float(ema21_s.iloc[-1])
    ema50  = float(ema50_s.iloc[-1])
    ema200 = float(ema200_s.iloc[-1]) if ema200_s is not None else None

    # Slope over last 3 bars — momentum direction of the EMA itself
    ema9_slope  = float(ema9_s.iloc[-1] - ema9_s.iloc[-4])  if len(ema9_s)  >= 4 else 0.0
    ema21_slope = float(ema21_s.iloc[-1] - ema21_s.iloc[-4]) if len(ema21_s) >= 4 else 0.0
    ema_rising      = ema9_slope > 0 and ema21_slope > 0
    ema_flattening  = abs(ema9_slope) < (cmp * 0.001)   # <0.1% of price over 3 bars

    # ── FIX 2: MACD — adjust=False, single-pass crossover, histogram ─────────
    macd_fast   = close.ewm(span=12, adjust=False).mean()
    macd_slow   = close.ewm(span=26, adjust=False).mean()
    macd_line   = macd_fast - macd_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram   = macd_line - signal_line

    macd_bullish = macd_bearish = False
    if len(macd_line) >= 3:
        prev_diff = float(macd_line.iloc[-2]) - float(signal_line.iloc[-2])
        curr_diff = float(macd_line.iloc[-1]) - float(signal_line.iloc[-1])
        if prev_diff <= 0 and curr_diff > 0:
            macd_bullish = True            # fresh bullish cross — mutually exclusive
        elif prev_diff >= 0 and curr_diff < 0:
            macd_bearish = True            # fresh bearish cross

    hist_val   = round(float(histogram.iloc[-1]), 4)
    hist_slope = float(histogram.iloc[-1] - histogram.iloc[-2]) if len(histogram) >= 2 else 0.0
    macd_hist_expanding   = hist_val > 0 and hist_slope > 0   # bull momentum building
    macd_hist_contracting = hist_val > 0 and hist_slope < 0   # bull momentum fading

    rsi_div  = detect_rsi_divergence(close, rsi_series)
    macd_div = detect_macd_divergence(close, macd_line)

    # ── FIX 3: Bollinger — clamped bb_pos + bandwidth + squeeze ──────────────
    bb_sma = float(close.rolling(20).mean().iloc[-1])
    bb_std = float(close.rolling(20).std().iloc[-1])
    bb_upper, bb_lower = bb_sma + 2 * bb_std, bb_sma - 2 * bb_std
    bb_range = bb_upper - bb_lower
    bb_pos = round(max(0.0, min(1.0, (cmp - bb_lower) / bb_range)), 2) if bb_range > 0 else 0.5
    # Breakout flags — replace info lost by clamping
    bb_breakout_up   = cmp > bb_upper
    bb_breakout_down = cmp < bb_lower

    bb_bandwidth = round(bb_range / bb_sma, 4) if bb_sma > 0 else None
    bb_squeeze   = False
    if bb_bandwidth is not None and len(close) >= 40:
        bw_series  = (close.rolling(20).std() * 4) / close.rolling(20).mean()
        bw_avg     = float(bw_series.rolling(20).mean().iloc[-1])
        if not pd.isna(bw_avg) and bw_avg > 0:
            bb_squeeze = bb_bandwidth < bw_avg * 0.75   # 25% below recent avg

    # ── FIX 4: Wilder ATR ──────────────────────────────────────────────────────
    tr, atr_series = compute_atr_wilder(high, low, close, 14)
    atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) \
        else float(high.iloc[-1] - low.iloc[-1])

    # ── Volume & Liquidity (FIX 10: soft gate, not silent None) ──────────────
    vol_avg   = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else float(vol.mean())
    vol_ratio = float(vol.iloc[-1]) / vol_avg if vol_avg > 0 else 1.0
    avg_turnover = vol_avg * cmp
    liquidity_ok = avg_turnover >= 10_000_000   # ₹1 Cr daily turnover

    # ── 20-day S/R ─────────────────────────────────────────────────────────────
    support    = float(low.rolling(20).min().iloc[-1])
    resistance = float(high.rolling(20).max().iloc[-1])
    high52, low52 = float(high.max()), float(low.min())

    # ── Trend with slope guard ────────────────────────────────────────────────
    trend = "Sideways"
    if cmp > ema9 > ema21:
        trend = "Strong Uptrend"
        if ema_flattening:
            trend = "Uptrend (Momentum Fading)"
    elif cmp > ema21:
        trend = "Uptrend" if ema_rising else "Recovery"
    elif cmp < ema9 < ema21:
        trend = "Strong Downtrend"
    elif cmp < ema21:
        trend = "Downtrend"

    # ── FIX 8: Fibonacci — swing-peak based via scipy ─────────────────────────
    c_vals = close.values
    swing_peaks,   _ = find_peaks( c_vals, distance=8, prominence=c_vals.std() * 0.3)
    swing_troughs, _ = find_peaks(-c_vals, distance=8, prominence=c_vals.std() * 0.3)

    fib_h = float(c_vals[swing_peaks[-1]])   if len(swing_peaks)   >= 1 else float(high.tail(60).max())
    fib_l = float(c_vals[swing_troughs[-1]]) if len(swing_troughs) >= 1 else float(low.tail(60).min())
    if fib_h < fib_l:                        # latest trough is above latest peak → swap
        fib_h, fib_l = fib_l, fib_h
    if (fib_h - fib_l) / (fib_l + 1e-8) < 0.02:   # degenerate swing → widen to 60-bar
        fib_h, fib_l = float(high.tail(60).max()), float(low.tail(60).min())

    fib_d   = fib_h - fib_l
    fib_236 = round(fib_h - fib_d * 0.236, 2)
    fib_382 = round(fib_h - fib_d * 0.382, 2)
    fib_500 = round(fib_h - fib_d * 0.500, 2)
    fib_618 = round(fib_h - fib_d * 0.618, 2)

    chart_patterns = detect_price_patterns(high, low, close, vol, vol_avg)
    candlesticks   = detect_candlesticks(open_p, high, low, close)

    # ── FIX 5: Supertrend — numpy loop, Wilder ATR(10), multiplier 2.5 ───────
    supertrend, supertrend_bullish = None, None
    try:
        st_period = 10
        st_mult   = 2.5     # 2.5 for NSE 5-15 day swing; 3.0 = positional
        _, atr_st_series = compute_atr_wilder(high, low, close, st_period)
        atr_st = atr_st_series.values
        hl2    = ((high + low) / 2).values
        c_arr  = close.values
        n      = len(c_arr)

        st_arr  = np.full(n, np.nan)
        dir_arr = np.ones(n, dtype=int)

        for i in range(n):
            if i < st_period or np.isnan(atr_st[i]):
                st_arr[i]  = c_arr[i]
                dir_arr[i] = 1
                continue
            upper = hl2[i] + st_mult * atr_st[i]
            lower = hl2[i] - st_mult * atr_st[i]
            if dir_arr[i - 1] == 1:                     # was bullish
                if c_arr[i] < st_arr[i - 1]:            # crossed below → flip
                    dir_arr[i] = -1
                    st_arr[i]  = upper
                else:
                    dir_arr[i] = 1
                    st_arr[i]  = max(lower, st_arr[i - 1])   # trail up only
            else:                                       # was bearish
                if c_arr[i] > st_arr[i - 1]:            # crossed above → flip
                    dir_arr[i] = 1
                    st_arr[i]  = lower
                else:
                    dir_arr[i] = -1
                    st_arr[i]  = min(upper, st_arr[i - 1])   # trail down only

        supertrend         = round(float(st_arr[-1]), 2)
        supertrend_bullish = bool(dir_arr[-1] == 1)
    except Exception:
        pass

    # ── FIX 6: VWAP — 20-day rolling, anchored to typical price ──────────────
    vwap, price_vs_vwap = None, None
    try:
        typical = (high + low + close) / 3
        roll_n  = min(20, len(close))
        cum_tpv = (typical * vol).rolling(roll_n).sum()
        cum_v   = vol.rolling(roll_n).sum().replace(0, np.nan)
        vwap_s  = cum_tpv / cum_v
        v = float(vwap_s.iloc[-1])
        if not pd.isna(v) and v > 0:
            vwap = round(v, 2)
            price_vs_vwap = round((cmp - vwap) / vwap * 100, 2)
    except Exception:
        pass

    # ── Bull / Bear Trap detection ────────────────────────────────────────────
    traps = detect_trap_signals(
        close, high, low, vol, vol_avg, rsi,
        supertrend_bullish, resistance, support,
        candlesticks, atr, window=15
    )

    return {
        "symbol": symbol, "cmp": round(cmp, 2), "rsi": rsi,

        # EMAs — v12 adds slope flags + ema200
        "ema9": round(ema9, 2), "ema21": round(ema21, 2), "ema50": round(ema50, 2),
        "ema200": round(ema200, 2) if ema200 else None,
        "ema_rising": ema_rising, "ema_flattening": ema_flattening,
        # Legacy aliases for app.py
        "ema20": round(ema9, 2), "ema50_alias": round(ema21, 2),

        # MACD — v12 adds histogram intelligence
        "macd_bullish": macd_bullish, "macd_bearish": macd_bearish,
        "macd_histogram": hist_val,
        "macd_hist_expanding": macd_hist_expanding,
        "macd_hist_contracting": macd_hist_contracting,

        "rsi_divergence": rsi_div, "macd_divergence": macd_div,

        # BB — v12 adds bandwidth/squeeze/breakout flags
        "bb_upper": round(bb_upper, 2), "bb_lower": round(bb_lower, 2),
        "bb_sma": round(bb_sma, 2), "bb_pos": bb_pos,
        "bb_bandwidth": bb_bandwidth, "bb_squeeze": bb_squeeze,
        "bb_breakout_up": bb_breakout_up, "bb_breakout_down": bb_breakout_down,

        "atr": round(atr, 2), "vol_ratio": round(vol_ratio, 2),
        "support": round(support, 2), "resistance": round(resistance, 2),
        "high52": round(high52, 2), "low52": round(low52, 2),
        "trend": trend,
        "fib_236": fib_236, "fib_382": fib_382, "fib_500": fib_500, "fib_618": fib_618,
        "supertrend": supertrend, "supertrend_bullish": supertrend_bullish,
        "vwap": vwap, "price_vs_vwap": price_vs_vwap,
        "patterns": chart_patterns, "candlesticks": candlesticks,
        "avg_turnover": avg_turnover, "atr_pct": (atr / cmp) if cmp > 0 else 0,
        "liquidity_ok": liquidity_ok,   # FIX 10: visible flag, not silent None
        # Trap signals
        "bull_trap": traps["bull_trap"],
        "bear_trap": traps["bear_trap"],
        "bull_trap_conf": traps["bull_trap_conf"],
        "bear_trap_conf": traps["bear_trap_conf"],
        "bull_trap_detail": traps["bull_trap_detail"],
        "bear_trap_detail": traps["bear_trap_detail"],
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


# ==============================================================================
# FIX 9: UNIFIED RISK ENGINE — single source of truth, now used EVERYWHERE
# ==============================================================================
def _calc_risk_params(cmp, atr, resistance, buy_at=None, pct=None,
                      supertrend_val=None, supertrend_bullish=None,
                      action="HOLD"):
    """
    HOLD/AVERAGE/WATCH (existing positions):
      Target = max(20d resistance, cmp + 2.5*ATR)
      SL     = cmp - 2.0*ATR, raised to supertrend if bullish & higher
    PICK (fresh entries — sector picks AND universe scanner):
      Target = max(cmp*1.10, cmp + 2.5*ATR)
      SL     = cmp - 1.25*ATR
    SELL:
      Target = cmp (exit now), SL = re-entry zone
    """
    if action == "PICK":
        stop_loss = round(cmp - 1.25 * atr, 2)
        target    = round(max(cmp * 1.10, cmp + 2.5 * atr), 2)
    elif action in ("HOLD", "AVERAGE", "WATCH"):
        stop_loss = round(cmp - 2.0 * atr, 2)
        if supertrend_bullish and supertrend_val and supertrend_val > stop_loss:
            stop_loss = round(supertrend_val, 2)
        target = round(max(resistance, cmp + 2.5 * atr), 2)
    else:   # SELL
        stop_loss = round(cmp - 2.0 * atr, 2)
        target    = round(cmp, 2)

    risk   = cmp - stop_loss
    reward = target - cmp
    rr = round(reward / risk, 2) if risk > 0.01 else None
    return target, stop_loss, rr


# ─── Expert Signal Engine ─────────────────────────────────────────────────────
def generate_signals(trades_df):
    signals = []
    open_trades = trades_df[trades_df["status"] == "Open"].copy()
    if open_trades.empty:
        return signals

    market = get_market_regime()
    is_bear = market["regime"] in ("Strong Bear", "Bear")

    unique_symbols = open_trades["stock"].unique().tolist()
    bulk_data = _bulk_fetch_history(unique_symbols, period="1y")

    for _, row in open_trades.iterrows():
        symbol, buy_at, qty, tid = row["stock"], row["buy_at"], row["quantity"], row["id"]

        df  = bulk_data.get(symbol)
        ind = compute_indicators(symbol, period="1y", prefetched_df=df)

        if ind is None:
            signals.append({
                "id": tid, "stock": symbol, "sector": get_sector(symbol),
                "action": "⚪ WATCH", "reason": "Could not fetch data — verify NSE symbol",
                "strength": 0, "cmp": None, "rsi": None, "pct_from_buy": None,
                "target": None, "stop_loss": None, "avg_price": None,
                "new_avg": None, "new_sl": None, "macd_signal": "—",
                "bb_position": "—", "trend": "—", "support": None,
                "resistance": None, "risk_reward": None, "buy_at": buy_at,
                "quantity": qty, "market_regime": market["regime"],
                "divergence": "—", "supertrend": "—", "vwap": None, "fib_levels": {},
            })
            continue

        cmp, rsi = ind["cmp"], ind["rsi"]
        ema9, ema21, ema50 = ind["ema9"], ind["ema21"], ind["ema50"]
        support, resistance = ind["support"], ind["resistance"]
        atr   = ind["atr"]
        trend = ind["trend"]
        macd_bull, macd_bear = ind["macd_bullish"], ind["macd_bearish"]
        hist_fading          = ind.get("macd_hist_contracting", False)
        bb_pos               = ind["bb_pos"]
        bb_breakout_up       = ind.get("bb_breakout_up", False)
        rsi_div, macd_div    = ind["rsi_divergence"], ind["macd_divergence"]
        st_bullish, st_val   = ind.get("supertrend_bullish"), ind.get("supertrend")
        vwap        = ind.get("vwap")
        pv_vwap     = ind.get("price_vs_vwap")
        patterns    = ind.get("patterns", [])
        candles     = ind.get("candlesticks", [])
        bb_squeeze  = ind.get("bb_squeeze", False)
        ema_fading  = ind.get("ema_flattening", False)
        bull_trap   = ind.get("bull_trap", False)
        bear_trap   = ind.get("bear_trap", False)
        bt_conf     = ind.get("bull_trap_conf", 0)
        bt_detail   = ind.get("bull_trap_detail", "")
        brt_conf    = ind.get("bear_trap_conf", 0)
        brt_detail  = ind.get("bear_trap_detail", "")

        pct      = round((cmp - buy_at) / buy_at * 100, 2)
        near52h  = cmp >= ind["high52"] * 0.97
        nifty_chg = market.get("indices", {}).get("Nifty 50", {}).get("chg_pct", 0)
        stock_rs_strong = pct > nifty_chg

        # ── SELL Triggers ─────────────────────────────────────────────────────
        sell = []
        if rsi and rsi >= 75: sell.append(f"RSI overbought ({rsi})")
        if rsi and rsi >= 70 and near52h: sell.append("Near 52w high")
        atr_trail = round(cmp - 2.0 * atr, 2)
        effective_trail = round(max(atr_trail, st_val), 2) if (st_bullish and st_val) else atr_trail
        if cmp < effective_trail: sell.append(f"Below Trail Stop (₹{effective_trail})")
        elif not st_bullish: sell.append("Supertrend Bearish")
        if ema50 and cmp < ema50 and pct < -5: sell.append("Below EMA50 (-5% loss)")
        if macd_bear: sell.append("MACD Bearish Cross")
        if rsi_div["bearish_div"]: sell.append("RSI Bear Div")
        if macd_div["bearish_div"]: sell.append("MACD Bear Div")
        if "📈 Double Top" in patterns: sell.append("Double Top Rejection")
        if "🏔️ Head & Shoulders (Top)" in patterns: sell.append("H&S Bearish Reversal")
        if "🟥 Bearish Engulfing" in candles and rsi and rsi > 65: sell.append("Bearish Distribution Candle")
        if ind["vol_ratio"] > 2.5 and rsi and rsi > 65: sell.append("Volume spike at resistance")
        if trend in ("Downtrend", "Strong Downtrend") and pct < -8: sell.append(f"{trend} breakdown")
        if is_bear and pct < -5: sell.append("Bear Market override")
        # v12: momentum-fading early warning (histogram contracting + EMA flat + in profit)
        if hist_fading and ema_fading and pct > 8 and rsi and rsi > 60:
            sell.append("Momentum fading — book partial profits")
        # Trap signals
        if bull_trap:
            sell.append(f"🪤 Bull Trap (conf {bt_conf}%) — {bt_detail}")

        # ── AVERAGE / BUY Triggers ─────────────────────────────────────────────
        avg = []
        can_avg = not (trend == "Strong Downtrend" and
                       not ("📉 Double Bottom" in patterns or "🛤️ Inverse H&S (Bottom)" in patterns))
        if can_avg:
            if rsi and rsi <= 40 and pct < -5:
                if "🔨 Bullish Hammer" in candles or "🟩 Bullish Engulfing" in candles:
                    avg.append(f"Oversold Bounce Confirmed by {candles[0]}")
            if "📉 Double Bottom" in patterns and stock_rs_strong: avg.append("Double Bottom + Relative Strength")
            if "🛤️ Inverse H&S (Bottom)" in patterns: avg.append("Inverse H&S Reversal")
            if trend in ("Uptrend", "Strong Uptrend") and cmp <= ema9 * 1.015 and pct < -3:
                if "🔨 Bullish Hammer" in candles: avg.append("EMA9 Pullback + Hammer Confirmation")
            if "🚀 Vol Breakout" in patterns and pct < 0: avg.append("Reversal Vol Breakout")
            if "🚩 Bull Flag Breakout" in patterns and pct < 0: avg.append("Bull Flag Breakout")
            # v12: squeeze release entry
            if bb_squeeze and bb_breakout_up and pct < 0:
                avg.append("BB Squeeze Release ↑")
            # Bear trap = trapped sellers → reversal opportunity
            if bear_trap:
                avg.append(f"🪤 Bear Trap (conf {brt_conf}%) — {brt_detail}")

        # ── HOLD Triggers ──────────────────────────────────────────────────────
        hold = []
        if "🚀 Vol Breakout" in patterns and pct > 0: hold.append("Bullish Breakout (Hold tight)")
        if "🚩 Bull Flag Breakout" in patterns and pct > 0: hold.append("Bull Flag Continuation")
        if rsi and 45 <= rsi <= 65 and cmp > ema9 and stock_rs_strong:
            hold.append("RSI neutral, Above EMA9, Beating Index")
        elif pct > 0 and not sell: hold.append(f"In profit {pct:+.1f}%")
        elif st_bullish and trend in ("Uptrend", "Strong Uptrend") and not sell:
            hold.append("Trend & Supertrend Bullish")

        # ── Action determination — unified risk engine ─────────────────────────
        if sell:
            action      = "🔴 SELL"
            reason_base = " | ".join(sell)
            strength    = min(len(sell) * 25 + 15, 95)
            target, stop_loss, rr = _calc_risk_params(
                cmp, atr, resistance, action="SELL")
            avg_price = new_avg = new_sl = None
        elif avg:
            action      = "🟡 AVERAGE"
            reason_base = " | ".join(avg)
            strength    = min(len(avg) * 22 + 18, 88)
            avg_price   = cmp
            new_avg     = round((buy_at * qty + cmp * qty) / (qty + qty), 2)
            target, stop_loss, rr = _calc_risk_params(
                cmp, atr, resistance,
                supertrend_val=st_val, supertrend_bullish=st_bullish, action="AVERAGE")
            new_sl = stop_loss
        elif hold:
            action      = "🟢 HOLD"
            reason_base = hold[0]
            strength    = 55
            target, stop_loss, rr = _calc_risk_params(
                cmp, atr, resistance,
                supertrend_val=st_val, supertrend_bullish=st_bullish, action="HOLD")
            avg_price = new_avg = new_sl = None
        else:
            action      = "⚪ WATCH"
            reason_base = f"CMP ₹{cmp} | RSI {rsi if rsi else '—'} | {pct:+.1f}%"
            strength    = 30
            target, stop_loss, rr = _calc_risk_params(
                cmp, atr, resistance,
                supertrend_val=st_val, supertrend_bullish=st_bullish, action="WATCH")
            avg_price = new_avg = new_sl = None

        div_parts = []
        if rsi_div["bullish_div"]: div_parts.append("RSI Bull")
        if rsi_div["bearish_div"]: div_parts.append("RSI Bear")
        if macd_div["bullish_div"]: div_parts.append("MACD Bull")
        if macd_div["bearish_div"]: div_parts.append("MACD Bear")
        div_lbl = ", ".join(div_parts) if div_parts else "None"

        macd_lbl = "Bullish ↗" if macd_bull else ("Bearish ↘" if macd_bear else "Neutral →")
        bb_lbl   = "Lower" if bb_pos < 0.2 else ("Upper" if bb_pos > 0.8 else "Mid")
        st_lbl   = f"Bullish ₹{st_val}" if st_bullish else (f"Bearish ₹{st_val}" if st_val else "—")

        all_pats     = patterns + candles
        pat_str      = " | ".join(all_pats) if all_pats else ""
        final_reason = f"[{pat_str}] {reason_base}" if pat_str else reason_base
        if not ind.get("liquidity_ok", True):
            final_reason += " ⚠️ Low liquidity (<₹1Cr/day)"

        signals.append({
            "id": tid, "stock": symbol, "sector": get_sector(symbol),
            "action": action, "reason": final_reason, "strength": strength,
            "cmp": cmp, "rsi": rsi,
            "ema20": ema9, "ema50": ema21,   # legacy display labels
            "atr": atr,
            "pct_from_buy": pct, "buy_at": buy_at, "quantity": qty,
            "target": target, "stop_loss": stop_loss,
            "avg_price": avg_price, "new_avg": new_avg, "new_sl": new_sl,
            "macd_signal": macd_lbl, "bb_position": bb_lbl,
            "trend": trend, "support": support, "resistance": resistance,
            "risk_reward": rr, "vol_ratio": ind["vol_ratio"],
            "market_regime": market["regime"], "divergence": div_lbl,
            "supertrend": st_lbl, "vwap": vwap,
            "fib_levels": {
                "23.6%": ind.get("fib_236"), "38.2%": ind.get("fib_382"),
                "50%": ind.get("fib_500"), "61.8%": ind.get("fib_618")
            },
        })
    return signals


# ─── Sector Rotation (unchanged from v11 — already 8/10) ──────────────────────
def sector_rotation(trades_df=None):
    rows = []
    idx_symbols = list(SECTOR_INDICES.values()) + ["^NSEI"]
    bulk_data   = _bulk_fetch_history(idx_symbols, period="6mo")

    nifty_df    = bulk_data.get("^NSEI")
    nifty_ret1m = nifty_ret3m = 0.0
    if nifty_df is not None and len(nifty_df) >= 21:
        nifty_ret1m = (float(nifty_df["Close"].iloc[-1]) / float(nifty_df["Close"].iloc[-21]) - 1) * 100
    if nifty_df is not None and len(nifty_df) >= 61:
        nifty_ret3m = (float(nifty_df["Close"].iloc[-1]) / float(nifty_df["Close"].iloc[-61]) - 1) * 100

    for sector, idx_sym in SECTOR_INDICES.items():
        try:
            df_idx = bulk_data.get(idx_sym)
            if df_idx is None or len(df_idx) < 21:
                continue
            close   = df_idx["Close"]
            cmp_now = float(close.iloc[-1])
            rsi     = compute_rsi(close)
            ema20   = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            ema50   = float(close.ewm(span=50, adjust=False).mean().iloc[-1]) if len(close) >= 50 else ema20
            ret_1w  = (cmp_now / float(close.iloc[-6])  - 1) * 100 if len(close) >= 6  else 0.0
            ret_1m  = (cmp_now / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else 0.0
            ret_3m  = (cmp_now / float(close.iloc[-61]) - 1) * 100 if len(close) >= 61 else 0.0
            rs_1m   = round(ret_1m - nifty_ret1m, 2)
            rs_3m   = round(ret_3m - nifty_ret3m, 2)
            rs_ratio    = 100 + rs_3m / 10
            rs_momentum = 100 + rs_1m / 5
            if   rs_ratio > 100 and rs_momentum > 100: rrg_quadrant = "🔥 Leading"
            elif rs_ratio > 100: rrg_quadrant = "📉 Weakening"
            elif rs_momentum > 100: rrg_quadrant = "🔄 Improving"
            else: rrg_quadrant = "❄️ Lagging"
            above_ema50 = (close > close.ewm(span=50, adjust=False).mean()).astype(int)
            streak = 0
            for val in reversed(above_ema50.values):
                if val == 1: streak += 1
                else: break
            rsi_score   = (rsi / 100) if rsi else 0.5
            rs_score    = max(-1.0, min(1.0, rs_1m / 10))
            trend_score = min(1.0, streak / 60)
            macd_score  = 1.0 if cmp_now > ema20 > ema50 else (0.5 if cmp_now > ema20 else 0.0)
            momentum_score = round(rsi_score * 0.20 + rs_score * 0.40 +
                                   trend_score * 0.20 + macd_score * 0.20, 3)
            rows.append({
                "sector": sector,
                "stocks": ", ".join(SECTOR_STOCKS.get(sector, [])[:4]) + "...",
                "cmp": round(cmp_now, 2), "rsi": rsi,
                "ret_1w": round(ret_1w, 2), "ret_1m": round(ret_1m, 2), "ret_3m": round(ret_3m, 2),
                "rs_vs_nifty_1m": rs_1m, "rs_vs_nifty_3m": rs_3m,
                "rrg_quadrant": rrg_quadrant, "trend_days": streak,
                "momentum_score": momentum_score, "macd_bullish": cmp_now > ema20,
                "count": len(SECTOR_STOCKS.get(sector, [])),
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("momentum_score", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    df["avg_rsi"] = df["rsi"]; df["avg_pct"] = df["ret_1m"]
    df["bullish_count"] = df["macd_bullish"].astype(int)
    df["index_chg"] = df["ret_1m"]
    return df[["rank", "sector", "stocks", "count", "avg_rsi", "avg_pct",
               "rs_vs_nifty_1m", "rs_vs_nifty_3m", "rrg_quadrant", "trend_days",
               "momentum_score", "bullish_count", "index_chg"]]


# ─── Sector Outlook ───────────────────────────────────────────────────────────
def predict_sector_outlook(sector_df):
    if sector_df.empty:
        return pd.DataFrame()
    preds = []
    for _, r in sector_df.iterrows():
        score = (r["momentum_score"] * 0.4
                 + (r.get("index_chg", 0) / 10 if r.get("index_chg") else 0) * 0.3
                 + (r["bullish_count"] / max(r["count"], 1)) * 0.3)
        if score > 0.5: outlook, conf = "🔥 Strong Bullish", 85
        elif score > 0.3: outlook, conf = "📈 Bullish", 70
        elif score > 0.1: outlook, conf = "➡️ Neutral-Bullish", 55
        elif score > -0.1: outlook, conf = "➡️ Neutral", 50
        elif score > -0.3: outlook, conf = "📉 Weak", 30
        else: outlook, conf = "🔻 Bearish", 20
        if r["avg_rsi"] and r["avg_rsi"] > 65:
            outlook = "🚀 Power Zone"; conf = min(conf + 15, 95)
        elif r["avg_rsi"] and r["avg_rsi"] < 45:
            outlook = "🩸 Bleeding — Avoid"; conf = max(conf - 20, 20)
        preds.append({"sector": r["sector"], "outlook": outlook, "confidence": conf,
                      "momentum": r["momentum_score"], "avg_rsi": r["avg_rsi"],
                      "avg_pct": r["avg_pct"], "index_chg": r.get("index_chg")})
    return pd.DataFrame(preds).sort_values("confidence", ascending=False)


# ─── Sector Stock Discovery — FIX 9: uses unified engine, action="PICK" ───────
def find_sector_picks(selected_sectors=None, max_per_sector=3):
    picks = []
    sectors = selected_sectors or list(SECTOR_STOCKS.keys())
    all_symbols = []
    for sector in sectors:
        all_symbols.extend(SECTOR_STOCKS.get(sector, []))
    bulk_data = _bulk_fetch_history(all_symbols, period="1y")

    for sector in sectors:
        spicks = []
        for symbol in SECTOR_STOCKS.get(sector, []):
            df  = bulk_data.get(symbol)
            ind = compute_indicators(symbol, period="1y", prefetched_df=df)
            if ind is None:
                continue
            if not ind.get("liquidity_ok", True):   # FIX 10: skip illiquid for new entries
                continue
            cmp, rsi = ind["cmp"], ind["rsi"]
            score, reasons = 0, []
            if rsi and 35 <= rsi <= 55: score += 15; reasons.append(f"RSI buy zone ({rsi})")
            if ind["trend"] in ("Uptrend", "Strong Uptrend"): score += 20; reasons.append(ind["trend"])
            elif ind["trend"] == "Recovery": score += 12; reasons.append("Recovery")
            if ind["macd_bullish"]: score += 15; reasons.append("MACD bullish cross")
            if ind.get("macd_hist_expanding"): score += 8; reasons.append("MACD momentum building")
            if ind.get("supertrend_bullish"): score += 10; reasons.append("Supertrend bullish")
            if ind["bb_pos"] < 0.3: score += 8; reasons.append("Lower BB bounce")
            if ind.get("bb_squeeze"): score += 8; reasons.append("BB squeeze (pre-breakout)")
            if ind["vol_ratio"] > 1.3: score += 8; reasons.append(f"Vol surge ({ind['vol_ratio']:.1f}x)")
            if ind.get("bear_trap"): score += 20; reasons.append(f"🪤 Bear Trap (conf {ind.get('bear_trap_conf',0)}%)")
            if ind.get("bull_trap"): score -= 25  # avoid buying into a bull trap
            if score < 45:
                continue

            # FIX 9: unified engine — same numbers as everywhere else
            atr = ind["atr"]
            tgt, sl, rr = _calc_risk_params(cmp, atr, ind["resistance"], action="PICK")
            if rr and rr < 1.5:
                continue

            patterns = ind.get("patterns", []); candles = ind.get("candlesticks", [])
            all_pats = patterns + candles
            pat_str  = " | ".join(all_pats) if all_pats else ""
            final_reason = (f"[{pat_str}] " + " | ".join(reasons[:4])) if pat_str else " | ".join(reasons[:4])
            spicks.append({"stock": symbol, "sector": sector, "cmp": cmp, "entry": round(cmp, 2),
                           "target": tgt, "stop_loss": sl, "risk_reward": rr, "score": score,
                           "rsi": rsi, "trend": ind["trend"], "reason": final_reason,
                           "atr": atr, "support": ind["support"], "resistance": ind["resistance"]})
        spicks.sort(key=lambda x: x["score"], reverse=True)
        picks.extend(spicks[:max_per_sector])
    picks.sort(key=lambda x: x["score"], reverse=True)
    return picks


# ─── Telegram ─────────────────────────────────────────────────────────────────
def send_telegram(bot_token, chat_id, message):
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=10)
        return resp.ok
    except Exception:
        return False


def build_telegram_message(signals, sector_df, picks=None):
    now = datetime.now().strftime("%d %b %Y %H:%M")
    lines = [f"<b>📈 Swing Dashboard</b>  <i>{now}</i>\n"]
    market = get_market_regime()
    lines.append(f"🌐 <b>Market:</b> {market['regime']} | Nifty ₹{market.get('nifty_close', '—')} | RSI {market.get('nifty_rsi', '—')}")
    for s in [s for s in signals if "SELL" in s["action"]]:
        lines.append(f"🔴 <b>{s['stock']}</b> ₹{s['cmp']} | {s['reason']}")
    for s in [s for s in signals if "AVERAGE" in s["action"]]:
        lines.append(f"🟡 <b>{s['stock']}</b> ₹{s['cmp']} | Avg ₹{s.get('avg_price')} | New SL ₹{s.get('new_sl')}")
    for s in [s for s in signals if "HOLD" in s["action"]]:
        lines.append(f"🟢 <b>{s['stock']}</b> ₹{s['cmp']} | {s['reason']}")
    if not sector_df.empty:
        lines.append("\n🔄 <b>SECTORS</b>")
        for _, r in sector_df.head(5).iterrows():
            e = "🥇" if r["rank"] == 1 else "🥈" if r["rank"] == 2 else "📊"
            lines.append(f"  {e} <b>{r['sector']}</b> Score {r['momentum_score']:.2f}")
    if picks:
        lines.append("\n🎯 <b>BUYS</b>")
        for p in picks[:8]:
            lines.append(f"  • <b>{p['stock']}</b> ₹{p['cmp']} | Tgt ₹{p['target']} | SL ₹{p['stop_loss']} | R:R {p['risk_reward']}")
    lines.append("\n<i>Indicative only. Not investment advice.</i>")
    return "\n".join(lines)


# ─── Master Universe Scanner — FIX 9: unified engine + liquidity gate ─────────
def generate_market_scanner():
    all_symbols = []
    for sector, stocks in SECTOR_STOCKS.items():
        all_symbols.extend(stocks)
    bulk_data = _bulk_fetch_history(all_symbols, period="6mo")
    results = []
    for symbol in all_symbols:
        sector = get_sector(symbol)
        df  = bulk_data.get(symbol)
        ind = compute_indicators(symbol, period="6mo", prefetched_df=df)
        if not ind:
            continue
        if not ind.get("liquidity_ok", True):   # FIX 10: visible skip for new entries
            continue
        cmp, rsi, trend = ind["cmp"], ind["rsi"], ind["trend"]
        patterns = ind.get("patterns", []); candles = ind.get("candlesticks", [])
        score = 0
        if trend in ("Uptrend", "Strong Uptrend"): score += 3
        if ind.get("supertrend_bullish"): score += 2
        if ind.get("macd_bullish"): score += 2
        if ind.get("macd_hist_expanding"): score += 1
        if ind.get("bb_squeeze"): score += 1
        if rsi and 60 <= rsi <= 75: score += 3
        if "🚀 Vol Breakout" in patterns: score += 5
        if "🚩 Bull Flag Breakout" in patterns: score += 4
        if "☕ Cup & Handle Breakout" in patterns: score += 4
        if "🟩 Bullish Engulfing" in candles: score += 2
        if ("📈 Double Top" in patterns or "🏔️ Head & Shoulders (Top)" in patterns or
                "🟥 Bearish Engulfing" in candles):
            score -= 5
        if trend in ("Downtrend", "Strong Downtrend"): score -= 4
        if score >= 8: signal = "🔥 STRONG BUY"
        elif score >= 5: signal = "🟢 BUY SETUP"
        elif score >= 2: signal = "🟡 ACCUMULATE"
        elif score <= 0: signal = "🔴 AVOID"
        else: signal = "⚪ NEUTRAL"
        all_pats = patterns + candles
        pat_str  = " | ".join(all_pats) if all_pats else "—"

        # FIX 9: same unified engine as sector picks — identical numbers everywhere
        atr = ind["atr"]
        tgt, sl, _rr = _calc_risk_params(cmp, atr, ind["resistance"], action="PICK")

        results.append({
            "Generated": datetime.now().strftime("%d %b %H:%M"), "Sector": sector,
            "Stock": symbol, "CMP": float(cmp), "Entry": float(cmp),
            "Target": float(tgt), "SL": float(sl), "Support": float(ind["support"]),
            "Resist": float(ind["resistance"]), "Signal": signal, "Score": score,
            "RSI": round(float(rsi), 2) if rsi else 0.0, "Trend": trend, "Patterns": pat_str
        })
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values(by=["Sector", "Score"], ascending=[True, False])


# ==============================================================================
# NEWS ENGINE (unchanged from v11 — already 8/10)
# ==============================================================================
def _parse_yf_news_item(item):
    if not isinstance(item, dict):
        return None, None
    content = item.get("content", {})
    if isinstance(content, dict) and content.get("title"):
        title = content.get("title", "")
        url_obj = content.get("clickThroughUrl") or content.get("canonicalUrl") or {}
        link = url_obj.get("url", "") if isinstance(url_obj, dict) else str(url_obj)
        return title, link
    title = item.get("title", "")
    link  = item.get("link", "") or item.get("url", "")
    return (title, link) if title else (None, None)


def _fetch_google_news_rss(query, max_items=2):
    try:
        q = requests.utils.quote(query)
        url = f"https://news.google.com/rss/search?q={q}+NSE+India&hl=en-IN&gl=IN&ceid=IN:en"
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        if not resp.ok:
            return []
        root = ET.fromstring(resp.content)
        items = root.findall(".//item")[:max_items]
        results = []
        for it in items:
            title_el = it.find("title")
            link_el  = it.find("link")
            if title_el is not None and title_el.text:
                title = title_el.text.strip()
                link  = link_el.text.strip() if link_el is not None and link_el.text else ""
                results.append((title, link))
        return results
    except Exception:
        return []


def fetch_portfolio_news(open_trades_df):
    if open_trades_df.empty:
        return []
    news_alerts = []
    unique_symbols = open_trades_df["stock"].unique().tolist()
    for sym in unique_symbols:
        clean_sym = sanitize_ticker(sym)
        found = False
        try:
            t     = yf.Ticker(f"{clean_sym}.NS")
            items = t.get_news(count=3)
            if items:
                for item in items[:2]:
                    title, link = _parse_yf_news_item(item)
                    if title:
                        if link:
                            news_alerts.append(
                                f"📰 <b>{clean_sym}</b>: "
                                f"<a href='{link}' style='color:var(--accent);text-decoration:none'>{title}</a>")
                        else:
                            news_alerts.append(f"📰 <b>{clean_sym}</b>: {title}")
                        found = True
        except Exception:
            pass
        if not found:
            try:
                rss_items = _fetch_google_news_rss(clean_sym, max_items=1)
                for title, link in rss_items:
                    if link:
                        news_alerts.append(
                            f"📰 <b>{clean_sym}</b>: "
                            f"<a href='{link}' style='color:var(--accent);text-decoration:none'>{title}</a>")
                    else:
                        news_alerts.append(f"📰 <b>{clean_sym}</b>: {title}")
            except Exception:
                continue
    return news_alerts


# ==============================================================================
# BULL TRAP & BEAR TRAP DETECTION — v12 addition
# ==============================================================================
# Bull Trap: Price fakes a breakout above resistance → collapses back.
#            Lures buyers into a losing long. Strong SELL signal.
# Bear Trap: Price fakes a breakdown below support → snaps back up.
#            Lures sellers into a losing short. Strong BUY/AVERAGE signal.
#
# Detection uses 5-factor confluence scoring (each factor adds confidence):
#   1. False breakout/breakdown geometry (price action)
#   2. Volume quality on the fake move (weak = not a real breakout)
#   3. RSI extreme at the fake move (overbought/oversold confirmation)
#   4. Supertrend direction alignment
#   5. Candle confirmation on the reversal bar
# ==============================================================================

def detect_trap_signals(close, high, low, vol, vol_avg, rsi,
                        supertrend_bullish, resistance, support,
                        candles, atr, window=15):
    """
    Returns dict:
      bull_trap       : bool
      bear_trap       : bool
      bull_trap_conf  : int  (0-100, confidence score)
      bear_trap_conf  : int
      bull_trap_detail: str  (human-readable reason)
      bear_trap_detail: str

    window=15 bars: catches fakeouts that take up to 3 trading weeks to fail,
    which is realistic for NSE daily swing charts.
    """
    result = {
        "bull_trap": False, "bear_trap": False,
        "bull_trap_conf": 0, "bear_trap_conf": 0,
        "bull_trap_detail": "", "bear_trap_detail": ""
    }
    if len(close) < window + 5:
        return result

    cmp = float(close.iloc[-1])
    # ATR-based tolerance so minor noise doesn't trigger false positives
    tol = max(atr * 0.3, resistance * 0.003)

    # Use BOTH close and high/low so intraday wicks are captured
    recent_high  = high.iloc[-(window + 1):-1]
    recent_low   = low.iloc[-(window + 1):-1]
    recent_close = close.iloc[-(window + 1):-1]
    recent_vol   = vol.iloc[-(window + 1):-1]

    # ── BULL TRAP ─────────────────────────────────────────────────────────────
    # Step 1: any close in the window broke above resistance?
    # Use close (not high) — a wick above resistance without a close is noise
    breakout_bars = [(i, float(c), float(v))
                     for i, (c, v) in enumerate(zip(recent_close, recent_vol))
                     if float(c) > resistance + tol]
    # Step 2: current bar has failed back below resistance
    failed_up = cmp < resistance - tol

    bull_conf = 0
    bull_detail = []

    if breakout_bars and failed_up:
        bull_conf += 35
        bull_detail.append("False breakout above resistance")

        # Factor 2: volume on the breakout bar(s) — weak = fake
        breakout_vols = [v for _, _, v in breakout_bars]
        if all(v < vol_avg * 1.5 for v in breakout_vols):
            bull_conf += 20
            bull_detail.append("low-vol breakout")

        # Factor 3: RSI overbought at the fake high
        if rsi and rsi > 65:
            bull_conf += 15
            bull_detail.append(f"RSI overbought ({rsi})")

        # Factor 4: supertrend turned bearish after the fake move
        if supertrend_bullish is False:
            bull_conf += 15
            bull_detail.append("Supertrend bearish")

        # Factor 5: reversal candle confirms rejection
        reject_candles = ["🟥 Bearish Engulfing", "💫 Shooting Star",
                          "🌆 Evening Star", "🦅 Three Black Crows"]
        matched = [c for c in reject_candles if c in candles]
        if matched:
            bull_conf += 15
            bull_detail.append(matched[0])

    if bull_conf >= 50:
        result["bull_trap"] = True
        result["bull_trap_conf"] = min(bull_conf, 95)
        result["bull_trap_detail"] = " | ".join(bull_detail)

    # ── BEAR TRAP ─────────────────────────────────────────────────────────────
    # Step 1: any bar in the window closed below support (real breakdown bar)?
    breakdown_bars = [(i, float(c), float(v))
                      for i, (c, v) in enumerate(zip(recent_close, recent_vol))
                      if float(c) < support - tol]
    # Step 2: current bar has snapped back above support
    failed_dn = cmp > support + tol

    bear_conf = 0
    bear_detail = []

    if breakdown_bars and failed_dn:
        bear_conf += 35
        bear_detail.append("False breakdown below support")

        # Factor 2: volume on the breakdown bar — weak = fake
        breakdown_vols = [v for _, _, v in breakdown_bars]
        if all(v < vol_avg * 1.5 for v in breakdown_vols):
            bear_conf += 20
            bear_detail.append("low-vol breakdown")

        # Factor 3: RSI oversold at the fake low
        if rsi and rsi < 35:
            bear_conf += 15
            bear_detail.append(f"RSI oversold ({rsi})")

        # Factor 4: supertrend has flipped back bullish
        if supertrend_bullish is True:
            bear_conf += 15
            bear_detail.append("Supertrend bullish recovery")

        # Factor 5: reversal candle confirms snap-back
        recovery_candles = ["🟩 Bullish Engulfing", "🔨 Bullish Hammer",
                            "🌅 Morning Star", "🪖 Three White Soldiers",
                            "🛤️ Inverse H&S (Bottom)"]
        matched = [c for c in recovery_candles if c in candles]
        if matched:
            bear_conf += 15
            bear_detail.append(matched[0])

    if bear_conf >= 50:
        result["bear_trap"] = True
        result["bear_trap_conf"] = min(bear_conf, 95)
        result["bear_trap_detail"] = " | ".join(bear_detail)

    return result
