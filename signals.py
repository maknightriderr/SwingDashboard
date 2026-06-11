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
    "Realty": "^CNXREALTY"
}

TRACKED_INDICES = {
    "Sensex": "^BSESN", # Added Sensex
    "Nifty 50": "^NSEI", "Bank Nifty": "^NSEBANK",
    "Nifty IT": "^CNXIT", "Nifty Fin": "^CNXFIN",
    "India VIX": "^INDIAVIX",
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

# ─── Chart Pattern Detection ──────────────────────────────────────────────────
def detect_price_patterns(high, low, close, vol, vol_avg):
    patterns = []
    if len(close) < 30: return patterns
    cmp = float(close.iloc[-1])

    recent_highs = high.iloc[-16:-1]
    recent_lows = low.iloc[-16:-1]
    range_pct = (recent_highs.max() - recent_lows.min()) / recent_lows.min()
    if range_pct < 0.08:
        if cmp > recent_highs.max() and float(vol.iloc[-1]) > vol_avg * 2:
            patterns.append("🚀 Vol Breakout")

    pole = close.iloc[-25:-8]
    flag = close.iloc[-8:-1]
    pole_gain = (pole.max() - pole.min()) / pole.min()
    flag_drop = (flag.max() - flag.min()) / flag.max()
    if pole_gain > 0.12 and flag_drop < 0.07 and flag.iloc[-1] < pole.max():
        if cmp > flag.max() and float(vol.iloc[-1]) > vol_avg * 1.5:
            patterns.append("🚩 Bull Flag Breakout")

    c_vals = close.values
    troughs, _ = find_peaks(-c_vals, distance=10)
    peaks, _ = find_peaks(c_vals, distance=10)

    if len(troughs) >= 2:
        t1, t2 = troughs[-2], troughs[-1]
        p1, p2 = c_vals[t1], c_vals[t2]
        if abs(p1 - p2) / p1 < 0.03 and p2 * 1.01 < cmp < p2 * 1.08:
            patterns.append("📉 Double Bottom")

    if len(peaks) >= 2:
        p1_idx, p2_idx = peaks[-2], peaks[-1]
        p1, p2 = c_vals[p1_idx], c_vals[p2_idx]
        if abs(p1 - p2) / p1 < 0.03 and p2 * 0.92 < cmp < p2 * 0.99:
            patterns.append("📈 Double Top")

    if len(peaks) >= 3:
        p1, p2, p3 = c_vals[peaks[-3]], c_vals[peaks[-2]], c_vals[peaks[-1]]
        if p2 > p1 and p2 > p3 and abs(p1 - p3) / p1 < 0.05:
            if cmp < p3 * 0.98:
                patterns.append("🏔️ Head & Shoulders (Top)")

    if len(troughs) >= 3:
        t1, t2, t3 = c_vals[troughs[-3]], c_vals[troughs[-2]], c_vals[troughs[-1]]
        if t2 < t1 and t2 < t3 and abs(t1 - t3) / t1 < 0.05:
            if cmp > t3 * 1.02:
                patterns.append("🛤️ Inverse H&S (Bottom)")

    return patterns

# ─── Candlestick Confirmation ─────────────────────────────────────────────────
def detect_candlesticks(open_p, high, low, close):
    candles = []
    if len(close) < 5: return candles
    
    o, h, l, c = float(open_p.iloc[-1]), float(high.iloc[-1]), float(low.iloc[-1]), float(close.iloc[-1])
    po, ph, pl, pc = float(open_p.iloc[-2]), float(high.iloc[-2]), float(low.iloc[-2]), float(close.iloc[-2])
    
    body = abs(c - o)
    total_range = h - l
    if total_range == 0: return candles

    lower_wick = o - l if c > o else c - l
    upper_wick = h - c if c > o else h - o
    
    if lower_wick > (body * 2) and upper_wick < (body * 0.5) and body > (total_range * 0.1):
        candles.append("🔨 Bullish Hammer")

    if pc < po and c > o:
        if o <= pc and c >= po and body > abs(pc - po):
            candles.append("🟩 Bullish Engulfing")

    if pc > po and c < o:
        if o >= pc and c <= po and body > abs(pc - po):
            candles.append("🟥 Bearish Engulfing")

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
            
            if nifty_rsi:
                if nifty_rsi > 75: conf = max(30, conf - 15)  
                elif nifty_rsi < 30: conf = max(30, conf - 15) 

    risk = "Low"
    if nifty_rsi and nifty_rsi > 70: risk = "High (Overbought)"
    elif nifty_rsi and nifty_rsi < 30: risk = "Low (Oversold)"
    elif nifty_rsi and nifty_rsi > 60: risk = "Moderate-High"

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
    df = prefetched_df
    if df is None:
        for suffix in [".NS", ".BO"]:
            df = _fetch_history(symbol + suffix, period=period, interval="1d")
            if df is not None: break

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

    rsi_series = compute_rsi_wilder(close, 14)
    rsi = round(float(rsi_series.iloc[-1]), 1) if not pd.isna(rsi_series.iloc[-1]) else None

    ema20 = float(close.ewm(span=20).mean().iloc[-1])
    ema50 = float(close.ewm(span=50).mean().iloc[-1])
    ema200 = float(close.ewm(span=200).mean().iloc[-1]) if len(close) >= 200 else None

    macd_line = close.ewm(span=12).mean() - close.ewm(span=26).mean()
    signal_line = macd_line.ewm(span=9).mean()
    macd_bullish = macd_bearish = False
    if len(macd_line) >= 4:
        for i in [-2, -1]:
            pd_ = float(macd_line.iloc[i-1]) - float(signal_line.iloc[i-1])
            cd_ = float(macd_line.iloc[i]) - float(signal_line.iloc[i])
            if pd_ <= 0 and cd_ > 0: macd_bullish = True
            elif pd_ >= 0 and cd_ < 0: macd_bearish = True

    rsi_div = detect_rsi_divergence(close, rsi_series)
    macd_div = detect_macd_divergence(close, macd_line)

    bb_sma = float(close.rolling(20).mean().iloc[-1])
    bb_std = float(close.rolling(20).std().iloc[-1])
    bb_upper, bb_lower = bb_sma + 2*bb_std, bb_sma - 2*bb_std
    bb_pos = (cmp - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5

    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else float(high.iloc[-1] - low.iloc[-1])

    # Volume Profiles & Institutional Liquidity Gate
    vol_avg = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else float(vol.mean())
    vol_ratio = float(vol.iloc[-1]) / vol_avg if vol_avg > 0 else 1.0

    # ──> THE LIQUIDITY GATE (Set to ₹1 Crore to prevent empty results) ────────
    avg_turnover = vol_avg * float(close.iloc[-1])
    if avg_turnover < 10000000:  
        return None
    # ──────────────────────────────────────────────────────────────────────────

    support = float(low.rolling(20).min().iloc[-1])
    resistance = float(high.rolling(20).max().iloc[-1])
    high52, low52 = float(high.max()), float(low.min())

    trend = "Sideways"
    if cmp > ema20 > ema50: trend = "Uptrend"
    elif cmp < ema20 < ema50: trend = "Downtrend"
    elif cmp > ema20 and ema20 < ema50: trend = "Recovery"
    elif cmp < ema20 and ema20 > ema50: trend = "Weakening"
    if ema200 and cmp > ema200 and trend == "Uptrend": trend = "Strong Uptrend"
    elif ema200 and cmp < ema200 and trend == "Downtrend": trend = "Strong Downtrend"

    fib_h, fib_l = float(high.tail(60).max()), float(low.tail(60).min())
    fib_d = fib_h - fib_l
    fib_236 = round(fib_h - fib_d*0.236, 2)
    fib_382 = round(fib_h - fib_d*0.382, 2)
    fib_500 = round(fib_h - fib_d*0.500, 2)
    fib_618 = round(fib_h - fib_d*0.618, 2)

    chart_patterns = detect_price_patterns(high, low, close, vol, vol_avg)
    candlesticks = detect_candlesticks(open_p, high, low, close)

    supertrend, supertrend_bullish = None, None
    try:
        hl2 = (high + low) / 2
        atr_st = tr.rolling(10).mean()
        upper = hl2 + 3 * atr_st  
        lower = hl2 - 3 * atr_st
        
        st = pd.Series(index=close.index, dtype=float)
        dir_ = pd.Series(index=close.index, dtype=int)
        
        for i in range(len(close)):
            if i < 10 or pd.isna(upper.iloc[i]):
                st.iloc[i] = float(close.iloc[i])
                dir_.iloc[i] = 1
                continue
            
            prev_st = float(st.iloc[i-1])
            prev_dir = dir_.iloc[i-1]
            curr_close = float(close.iloc[i])
            
            if prev_dir == 1:
                if curr_close < prev_st:
                    dir_.iloc[i] = -1
                    st.iloc[i] = float(upper.iloc[i])
                else:
                    dir_.iloc[i] = 1
                    st.iloc[i] = max(float(lower.iloc[i]), prev_st)
            else:
                if curr_close > prev_st:
                    dir_.iloc[i] = 1
                    st.iloc[i] = float(lower.iloc[i])
                else:
                    dir_.iloc[i] = -1
                    st.iloc[i] = min(float(upper.iloc[i]), prev_st)
                    
        supertrend = round(float(st.iloc[-1]), 2)
        supertrend_bullish = dir_.iloc[-1] == 1
    except Exception: 
        pass

    vwap = None
    try:
        typ = (high + low + close) / 3
        vwap = round(float((typ * vol).tail(5).sum() / vol.tail(5).sum()), 2)
    except Exception: pass

    return {
        "symbol": symbol, "cmp": round(cmp, 2), "rsi": rsi,
        "ema20": round(ema20, 2), "ema50": round(ema50, 2),
        "ema200": round(ema200, 2) if ema200 else None,
        "macd_bullish": macd_bullish, "macd_bearish": macd_bearish,
        "rsi_divergence": rsi_div, "macd_divergence": macd_div,
        "bb_upper": round(bb_upper, 2), "bb_lower": round(bb_lower, 2),
        "bb_sma": round(bb_sma, 2), "bb_pos": round(bb_pos, 2),
        "atr": round(atr, 2), "vol_ratio": round(vol_ratio, 2),
        "support": round(support, 2), "resistance": round(resistance, 2),
        "high52": round(high52, 2), "low52": round(low52, 2),
        "trend": trend,
        "fib_236": fib_236, "fib_382": fib_382, "fib_500": fib_500, "fib_618": fib_618,
        "supertrend": supertrend, "supertrend_bullish": supertrend_bullish, "vwap": vwap,
        "patterns": chart_patterns,
        "candlesticks": candlesticks,
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
                "action": "⚪ WATCH", "reason": "Could not fetch data",
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

# ─── Sector Rotation ──────────────────────────────────────────────────────────
def sector_rotation(trades_df=None): 
    rows = []
    idx_symbols = list(SECTOR_INDICES.values())
    bulk_data = _bulk_fetch_history(idx_symbols, period="3mo")

    for sector, idx_sym in SECTOR_INDICES.items():
        try:
            df_idx = bulk_data.get(idx_sym)
            if df_idx is not None and len(df_idx) >= 20:
                close = df_idx["Close"]
                cmp = float(close.iloc[-1])
                rsi = compute_rsi(close)
                ema20 = float(close.ewm(span=20).mean().iloc[-1])
                pct_chg = round((cmp / float(close.iloc[0]) - 1) * 100, 2)
                macd_bullish = cmp > ema20
                
                rows.append({
                    "sector": sector, 
                    "stocks": ", ".join(SECTOR_STOCKS.get(sector, [])[:4]) + "...",
                    "rsi": rsi, "pct_chg": pct_chg, "cmp": cmp, 
                    "macd_bullish": macd_bullish, 
                    "count": len(SECTOR_STOCKS.get(sector, []))
                })
        except Exception:
            pass
    
    df = pd.DataFrame(rows)
    if df.empty: return df

    df["rsi_score"] = df["rsi"].fillna(50) / 100
    max_pct = df["pct_chg"].abs().max() or 1
    df["pct_score"] = df["pct_chg"].fillna(0) / max_pct
    
    df["momentum_score"] = round((df["rsi_score"] * 0.4) + (df["pct_score"] * 0.6), 3)
    df = df.sort_values("momentum_score", ascending=False)
    df["rank"] = range(1, len(df)+1)
    df["avg_rsi"] = df["rsi"]
    df["avg_pct"] = df["pct_chg"]
    df["bullish_count"] = np.where(df["macd_bullish"], 1, 0)
    df["index_chg"] = df["pct_chg"]
    
    return df[["rank", "sector", "stocks", "count", "avg_rsi", "avg_pct", "momentum_score", "bullish_count", "index_chg"]]

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
        elif score > -0.3: outlook, conf = "📉 Weak", 40
        else: outlook, conf = "🔻 Bearish", 30
        if r["avg_rsi"] and r["avg_rsi"] > 70: outlook = "⚠️ Overbought"
        elif r["avg_rsi"] and r["avg_rsi"] < 35: outlook = "🟢 Oversold — Bounce"; conf = min(conf+10, 95)
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
            sl = round(max(ind["support"], cmp - 2*ind["atr"]), 2)
            tgt = round(max(ind["resistance"], cmp + 3*ind["atr"]), 2)
            risk, reward = cmp - sl, tgt - cmp
            rr = round(reward/risk, 2) if risk > 0 else None
            
            if rr and rr < 1.5: continue
            
            patterns = ind.get("patterns", [])
            candles = ind.get("candlesticks", [])
            all_pats = patterns + candles
            pat_str = " | ".join(all_pats) if all_pats else ""
            final_reason = f"[{pat_str}] " + " | ".join(reasons[:4]) if pat_str else " | ".join(reasons[:4])

            spicks.append({"stock": symbol, "sector": sector, "cmp": cmp, "entry": round(cmp,2),
                           "target": tgt, "stop_loss": sl, "risk_reward": rr, "score": score,
                           "rsi": rsi, "trend": ind["trend"], "reason": final_reason,
                           "atr": ind["atr"], "support": ind["support"], "resistance": ind["resistance"]})
                           
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
        
        # Confluence Scoring Engine for New Entries
        score = 0
        
        if trend in ("Uptrend", "Strong Uptrend"): score += 2
        if ind.get("supertrend_bullish"): score += 1
        if ind.get("macd_bullish"): score += 1
        if rsi and 40 <= rsi <= 60: score += 1
        
        if "🚀 Vol Breakout" in patterns: score += 4
        if "🚩 Bull Flag Breakout" in patterns: score += 3
        if "📉 Double Bottom" in patterns: score += 3
        if "🛤️ Inverse H&S (Bottom)" in patterns: score += 3
        if "🔨 Bullish Hammer" in candles: score += 2
        if "🟩 Bullish Engulfing" in candles: score += 2
        
        if "📈 Double Top" in patterns or "🏔️ Head & Shoulders (Top)" in patterns or "🟥 Bearish Engulfing" in candles:
            score -= 4
        if trend in ("Downtrend", "Strong Downtrend"):
            score -= 2
            
        # Signal Generation
        if score >= 6: signal = "🔥 STRONG BUY"
        elif score >= 4: signal = "🟢 BUY SETUP"
        elif score >= 2: signal = "🟡 ACCUMULATE"
        elif score <= 0: signal = "🔴 AVOID"
        else: signal = "⚪ NEUTRAL"
        
        all_pats = patterns + candles
        pat_str = " | ".join(all_pats) if all_pats else "—"
        
        # Calculate strict risk metrics for Universe Scan Output
        atr = ind["atr"]
        sup, res = ind["support"], ind["resistance"]
        sl = round(max(sup, cmp - 2*atr), 2)
        tgt = round(max(res, cmp + 3*atr), 2)
        
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
            "Patterns": pat_str
        })
        
    return pd.DataFrame(results).sort_values(by=["Sector", "Score"], ascending=[True, False])
