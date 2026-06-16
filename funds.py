"""
funds.py — Mutual Fund & ETF module (FULLY SEPARATE from stock/signals logic)
================================================================================
This module is self-contained and shares NO state with signals.py. It powers two
dashboard pages: a Mutual Fund tracker and an ETF tracker.

DATA SOURCES (both free, no API key):
  • Mutual Funds → AMFI India via mfapi.in  (daily NAV, full history, reliable)
  • ETFs         → Yahoo Finance via yfinance (they trade like stocks)

WHY SEPARATE FROM STOCKS:
  • Mutual funds have only daily NAV — no intraday OHLC, no volume, no tick data,
    so RSI/SMC/Supertrend/Trap detection are meaningless for them.
  • ETFs *can* run technicals, but the user wants them tracked apart from the
    stock portfolio, so this module keeps its own data + UI.

PUBLIC API:
  MUTUAL FUNDS
    search_mf(query)                  -> list[{schemeCode, schemeName}]
    get_mf_nav(scheme_code)           -> dict (latest NAV + meta)
    get_mf_history(scheme_code)       -> DataFrame[date, nav]
    get_mf_returns(scheme_code)       -> dict (1M/3M/6M/1Y/3Y % returns)
    mf_summary(scheme_code)           -> dict (everything combined for a card)
    compare_mfs([codes])              -> DataFrame side-by-side

  ETFs
    ETF_UNIVERSE                      -> dict {symbol: name}
    get_etf_quote(symbol)             -> dict (CMP, day chg, 52w, returns)
    get_etf_history(symbol, period)   -> DataFrame
    scan_etfs()                       -> DataFrame of all ETFs with returns
    etf_summary(symbol)               -> dict for a card
"""

import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ──────────────────────────────────────────────────────────────────────────────
#  SIMPLE TIME CACHE  (module-local; nothing shared with signals.py)
# ──────────────────────────────────────────────────────────────────────────────
_CACHE = {}
_CACHE_TTL = 1800   # 30 min — NAVs update once daily, so this is plenty fresh


def _cache_get(key):
    v = _CACHE.get(key)
    if v and (time.time() - v[1]) < _CACHE_TTL:
        return v[0]
    return None


def _cache_set(key, val):
    _CACHE[key] = (val, time.time())
    return val


_HEADERS = {"User-Agent": "Mozilla/5.0 (SwingDashboard MF/ETF module)"}


# ==============================================================================
#  MUTUAL FUNDS  —  via AMFI / mfapi.in
# ==============================================================================
#
# mfapi.in endpoints:
#   https://api.mfapi.in/mf                      → full scheme list (search)
#   https://api.mfapi.in/mf/{schemeCode}         → full NAV history + meta
#   https://api.mfapi.in/mf/{schemeCode}/latest  → latest NAV only
#
_MF_LIST_URL   = "https://api.mfapi.in/mf"
_MF_SCHEME_URL = "https://api.mfapi.in/mf/{code}"
_MF_LATEST_URL = "https://api.mfapi.in/mf/{code}/latest"

# In-memory cache of the full scheme list (large, ~45k schemes) for searching.
_MF_SCHEME_LIST = None


def _load_mf_list():
    """Download (once) and cache the full AMFI scheme list for searching."""
    global _MF_SCHEME_LIST
    if _MF_SCHEME_LIST is not None:
        return _MF_SCHEME_LIST
    cached = _cache_get("mf_list")
    if cached is not None:
        _MF_SCHEME_LIST = cached
        return cached
    for _attempt in range(3):
        try:
            r = requests.get(_MF_LIST_URL, headers=_HEADERS, timeout=20)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data:
                    _MF_SCHEME_LIST = data
                    _cache_set("mf_list", data)
                    return data
        except Exception:
            time.sleep(0.5)
    return []


def search_mf(query, limit=25):
    """Search mutual funds by name. Returns list of {schemeCode, schemeName}."""
    query = (query or "").strip().lower()
    if len(query) < 3:
        return []
    schemes = _load_mf_list()
    if not schemes:
        return []
    # Token-based match: every query word must appear in the scheme name.
    words = query.split()
    results = []
    for s in schemes:
        name = str(s.get("schemeName", "")).lower()
        if all(w in name for w in words):
            results.append({
                "schemeCode": s.get("schemeCode"),
                "schemeName": s.get("schemeName"),
            })
            if len(results) >= limit:
                break
    return results


def get_mf_history(scheme_code):
    """Full NAV history for a scheme → DataFrame[date(datetime), nav(float)]
    sorted ascending by date. Returns empty DataFrame on failure."""
    key = f"mf_hist_{scheme_code}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    for _attempt in range(3):
        try:
            r = requests.get(_MF_SCHEME_URL.format(code=scheme_code),
                             headers=_HEADERS, timeout=20)
            if r.status_code == 200:
                payload = r.json()
                data = payload.get("data", [])
                if data:
                    df = pd.DataFrame(data)
                    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
                    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y",
                                                errors="coerce")
                    df = df.dropna(subset=["date", "nav"]).sort_values("date")
                    df = df.reset_index(drop=True)
                    # Attach meta as attrs so callers can read fund house etc.
                    meta = payload.get("meta", {})
                    df.attrs["meta"] = meta
                    return _cache_set(key, df)
        except Exception:
            time.sleep(0.5)
    return pd.DataFrame(columns=["date", "nav"])


def get_mf_nav(scheme_code):
    """Latest NAV + metadata for a scheme. Returns dict."""
    key = f"mf_nav_{scheme_code}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    for _attempt in range(3):
        try:
            r = requests.get(_MF_LATEST_URL.format(code=scheme_code),
                             headers=_HEADERS, timeout=15)
            if r.status_code == 200:
                payload = r.json()
                meta = payload.get("meta", {})
                data = payload.get("data", [])
                latest = data[0] if data else {}
                out = {
                    "scheme_code": scheme_code,
                    "scheme_name": meta.get("scheme_name", ""),
                    "fund_house":  meta.get("fund_house", ""),
                    "scheme_type": meta.get("scheme_type", ""),
                    "scheme_category": meta.get("scheme_category", ""),
                    "nav":         float(latest.get("nav", 0) or 0),
                    "nav_date":    latest.get("date", ""),
                }
                return _cache_set(key, out)
        except Exception:
            time.sleep(0.5)
    return {"scheme_code": scheme_code, "scheme_name": "", "nav": 0, "nav_date": ""}


def _pct_return(hist, days):
    """% change in NAV over the last `days` calendar days using the history df."""
    if hist is None or hist.empty or len(hist) < 2:
        return None
    latest_date = hist["date"].iloc[-1]
    latest_nav  = float(hist["nav"].iloc[-1])
    target_date = latest_date - timedelta(days=days)
    # Find the closest available NAV on/before target_date
    past = hist[hist["date"] <= target_date]
    if past.empty:
        # Not enough history for this window
        return None
    past_nav = float(past["nav"].iloc[-1])
    if past_nav <= 0:
        return None
    return round((latest_nav / past_nav - 1) * 100, 2)


def get_mf_returns(scheme_code):
    """Trailing returns for 1M/3M/6M/1Y/3Y/5Y. Values are % or None if N/A."""
    hist = get_mf_history(scheme_code)
    if hist is None or hist.empty:
        return {k: None for k in ["1M", "3M", "6M", "1Y", "3Y", "5Y"]}
    return {
        "1M": _pct_return(hist, 30),
        "3M": _pct_return(hist, 91),
        "6M": _pct_return(hist, 182),
        "1Y": _pct_return(hist, 365),
        "3Y": _annualised(_pct_return(hist, 365 * 3), 3),
        "5Y": _annualised(_pct_return(hist, 365 * 5), 5),
    }


def _annualised(total_pct, years):
    """Convert a total (cumulative) % return over `years` into annualised CAGR %."""
    if total_pct is None:
        return None
    try:
        growth = 1 + total_pct / 100.0
        if growth <= 0:
            return None
        cagr = (growth ** (1.0 / years) - 1) * 100
        return round(cagr, 2)
    except Exception:
        return None


def compute_mf_rating(returns, hist=None):
    """Produce a long-term RATING for a mutual fund (NOT a trade signal).

    Mutual funds are long-term instruments — timing buy/sell on NAV moves is
    inappropriate and can be harmful. Instead this gives an investment-quality
    rating based on trailing returns, consistency, and momentum:
      ⭐ STRONG    — strong, consistent multi-period returns
      ✅ GOOD      — solid performer
      ➖ AVERAGE   — middling
      ⚠️ WEAK      — underperforming

    Plus a SIP-oriented note. Returns dict: {rating, stars, score, note, action}.
    """
    out = {"rating": "—", "stars": 0, "score": 0, "note": "Insufficient history",
           "action": "—"}
    if not returns:
        return out

    r1y = returns.get("1Y")
    r3y = returns.get("3Y")
    r5y = returns.get("5Y")
    r6m = returns.get("6M")
    r3m = returns.get("3M")

    score = 0
    pts = 0   # count of available metrics so we can normalise

    # Long-term CAGR is the biggest weight (what matters for MFs)
    if r3y is not None:
        pts += 1
        if   r3y >= 18: score += 30
        elif r3y >= 12: score += 22
        elif r3y >= 8:  score += 12
        elif r3y >= 0:  score += 4
        else:           score -= 10
    if r5y is not None:
        pts += 1
        if   r5y >= 16: score += 25
        elif r5y >= 11: score += 18
        elif r5y >= 7:  score += 10
        elif r5y >= 0:  score += 3
        else:           score -= 8
    if r1y is not None:
        pts += 1
        if   r1y >= 20: score += 20
        elif r1y >= 12: score += 14
        elif r1y >= 5:  score += 7
        elif r1y >= 0:  score += 2
        else:           score -= 8
    # Recent momentum (smaller weight)
    if r6m is not None:
        pts += 1
        if   r6m >= 10: score += 12
        elif r6m >= 3:  score += 6
        elif r6m < -5:  score -= 6
    if r3m is not None:
        pts += 1
        if   r3m >= 6:  score += 8
        elif r3m < -6:  score -= 5

    if pts == 0:
        return out

    # Normalise to a 0-100-ish scale (max possible ≈ 95)
    norm = max(0, min(100, int(score / 95 * 100)))

    if   norm >= 70:
        rating, stars, action = "STRONG", 5, "Good for SIP / lump-sum (long-term)"
    elif norm >= 50:
        rating, stars, action = "GOOD", 4, "Suitable for SIP accumulation"
    elif norm >= 32:
        rating, stars, action = "AVERAGE", 3, "Hold if invested; compare peers before adding"
    elif norm >= 18:
        rating, stars, action = "WEAK", 2, "Review vs category leaders"
    else:
        rating, stars, action = "POOR", 1, "Underperforming — consider better-rated peers"

    # Build a short rationale note
    bits = []
    if r3y is not None: bits.append(f"3Y CAGR {r3y}%")
    if r1y is not None: bits.append(f"1Y {r1y}%")
    if r6m is not None: bits.append(f"6M {r6m}%")
    note = " · ".join(bits) if bits else "Limited data"

    return {"rating": rating, "stars": stars, "score": norm,
            "note": note, "action": action}


def mf_summary(scheme_code):
    """Everything needed to render a fund card: NAV, meta, returns, 52w range, rating."""
    nav = get_mf_nav(scheme_code)
    hist = get_mf_history(scheme_code)
    returns = get_mf_returns(scheme_code)

    high52 = low52 = None
    if hist is not None and not hist.empty:
        last_year = hist[hist["date"] >= (hist["date"].iloc[-1] - timedelta(days=365))]
        if not last_year.empty:
            high52 = round(float(last_year["nav"].max()), 2)
            low52  = round(float(last_year["nav"].min()), 2)

    rating = compute_mf_rating(returns, hist)

    return {
        **nav,
        "returns": returns,
        "high52": high52,
        "low52": low52,
        "rating": rating,
        "has_history": hist is not None and not hist.empty,
    }


def compare_mfs(scheme_codes):
    """Side-by-side comparison table for multiple funds.
    Returns a DataFrame with one row per fund."""
    rows = []
    for code in scheme_codes:
        s = mf_summary(code)
        r = s.get("returns", {})
        rating = s.get("rating", {})
        rows.append({
            "Fund": s.get("scheme_name", str(code))[:45],
            "Rating": f"{'⭐'*rating.get('stars',0)} {rating.get('rating','—')}",
            "Category": s.get("scheme_category", ""),
            "NAV": s.get("nav"),
            "1M %": r.get("1M"), "3M %": r.get("3M"), "6M %": r.get("6M"),
            "1Y %": r.get("1Y"), "3Y %": r.get("3Y"), "5Y %": r.get("5Y"),
        })
    return pd.DataFrame(rows)


# ==============================================================================
#  ETFs  —  via Yahoo Finance (they trade like stocks)
# ==============================================================================
#
# Curated list of the most-traded NSE ETFs. symbol → display name.
# These are appended with .NS for Yahoo.
#
ETF_UNIVERSE = {
    # Broad equity index
    "NIFTYBEES":   "Nippon Nifty 50 ETF",
    "SETFNIF50":   "SBI Nifty 50 ETF",
    "ICICINIFTY":  "ICICI Pru Nifty 50 ETF",
    "UTINIFTETF":  "UTI Nifty 50 ETF",
    "HDFCNIFTY":   "HDFC Nifty 50 ETF",
    "JUNIORBEES":  "Nippon Nifty Next 50 ETF",
    "NEXT50":      "ICICI Pru Nifty Next 50 ETF",
    # Bank / sectoral
    "BANKBEES":    "Nippon Nifty Bank ETF",
    "SETFNIFBK":   "SBI Nifty Bank ETF",
    "ICICIBANKN":  "ICICI Pru Nifty Bank ETF",
    "ITBEES":      "Nippon Nifty IT ETF",
    "ICICITECH":   "ICICI Pru Nifty IT ETF",
    "PSUBNKBEES":  "Nippon Nifty PSU Bank ETF",
    "PHARMABEES":  "Nippon Nifty Pharma ETF",
    "AUTOBEES":    "Nippon Nifty Auto ETF",
    "CONSUMBEES":  "Nippon Nifty Consumption ETF",
    "INFRABEES":   "Nippon Nifty Infra ETF",
    # Broad market / strategy
    "MOM100":      "Motilal Nifty Midcap 100 ETF",
    "MIDCAPETF":   "ICICI Pru Nifty Midcap 150 ETF",
    "MOSMALL250":  "Motilal Nifty Smallcap 250 ETF",
    "ICICIB22":    "Bharat 22 ETF (ICICI)",
    "CPSEETF":     "CPSE ETF",
    "MAFANG":      "Mirae NYSE FANG+ ETF",
    "MON100":      "Motilal Nasdaq 100 ETF",
    "MOM50":       "Motilal Nifty 50 ETF",
    "ALPHA":       "Nippon Nifty Alpha Low Vol ETF",
    "LOWVOL":      "ICICI Nifty Low Vol 30 ETF",
    "MOVALUE":     "Motilal Nifty 200 Momentum 30 ETF",
    # Gold / silver / commodity
    "GOLDBEES":    "Nippon Gold ETF",
    "GOLDSHARE":   "UTI Gold ETF",
    "SETFGOLD":    "SBI Gold ETF",
    "HDFCGOLD":    "HDFC Gold ETF",
    "AXISGOLD":    "Axis Gold ETF",
    "SILVERBEES":  "Nippon Silver ETF",
    "SILVER":      "ICICI Pru Silver ETF",
    # Debt / liquid
    "LIQUIDBEES":  "Nippon Liquid ETF",
    "LIQUIDETF":   "DSP Liquid ETF",
}


def _etf_yf():
    """Lazy import yfinance so this module loads even if yfinance is missing."""
    import yfinance as yf
    return yf


def get_etf_history(symbol, period="1y"):
    """OHLC history for an ETF. Returns DataFrame or empty on failure."""
    key = f"etf_hist_{symbol}_{period}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    yf = _etf_yf()
    clean = str(symbol).upper().strip().replace(".NS", "")
    for _attempt in range(2):
        # Method 1: Ticker.history
        try:
            t = yf.Ticker(clean + ".NS")
            df = t.history(period=period, interval="1d", auto_adjust=False)
            if df is not None and not df.empty and "Close" in df.columns:
                return _cache_set(key, df)
        except Exception:
            pass
        # Method 2: download fallback
        try:
            df = yf.download(clean + ".NS", period=period, interval="1d",
                             auto_adjust=False, progress=False, threads=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if "Close" in df.columns:
                    return _cache_set(key, df)
        except Exception:
            pass
        time.sleep(0.3)
    return pd.DataFrame()


def _etf_fast_price(symbol):
    """Live last price for an ETF via fast_info (most accurate)."""
    yf = _etf_yf()
    clean = str(symbol).upper().strip().replace(".NS", "")
    try:
        t = yf.Ticker(clean + ".NS")
        fi = t.fast_info
        for key in ("last_price", "lastPrice", "regularMarketPrice"):
            for getter in (lambda: fi.get(key) if hasattr(fi, "get") else None,
                           lambda: getattr(fi, key, None),
                           lambda: fi[key] if hasattr(fi, "__getitem__") else None):
                try:
                    v = getter()
                    if v is not None and not pd.isna(v) and float(v) > 0:
                        return float(v)
                except Exception:
                    pass
    except Exception:
        pass
    return None


def _hist_return(hist, days):
    """% return over `days` using an OHLC history frame (Close column)."""
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None
    closes = hist["Close"].dropna()
    if len(closes) < 2:
        return None
    latest = float(closes.iloc[-1])
    # approximate trading-day count (≈ days * 5/7)
    lookback = int(days * 5 / 7)
    if lookback >= len(closes):
        past = float(closes.iloc[0])
    else:
        past = float(closes.iloc[-lookback - 1])
    if past <= 0:
        return None
    return round((latest / past - 1) * 100, 2)


def _ema(series, span):
    """Exponential moving average of a pandas Series."""
    return series.ewm(span=span, adjust=False).mean()


def _rsi(closes, period=14):
    """Wilder's RSI for the last value. Returns float or None."""
    if closes is None or len(closes) < period + 1:
        return None
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    if pd.isna(val):
        return 100.0 if avg_loss.iloc[-1] == 0 else 50.0
    return round(float(val), 1)


def compute_etf_signal(symbol, hist=None):
    """Generate a BUY / HOLD / SELL signal for an ETF using real technicals.

    ETFs trade like stocks, so this uses genuine trend + momentum logic:
      • Price vs 50-DMA and 200-DMA (trend)
      • Golden/death cross (50 vs 200)
      • RSI (momentum / overbought-oversold)
      • Position within the 52-week range
      • 3-month return trend
    Returns dict: {signal, score (-100..100), confidence, reasons[], rsi, dma50, dma200}.
    """
    if hist is None:
        hist = get_etf_history(symbol, period="1y")
    out = {"signal": "NO DATA", "score": 0, "confidence": "—",
           "reasons": [], "rsi": None, "dma50": None, "dma200": None,
           "entry": None, "target": None, "stop_loss": None}
    if hist is None or hist.empty or "Close" not in hist.columns:
        return out
    closes = hist["Close"].dropna()
    if len(closes) < 50:
        return out

    cmp     = float(closes.iloc[-1])
    dma50   = float(_ema(closes, 50).iloc[-1])
    dma200  = float(_ema(closes, 200).iloc[-1]) if len(closes) >= 200 else None
    rsi     = _rsi(closes)
    hi52    = float(closes.max())
    lo52    = float(closes.min())
    ret_3m  = _hist_return(hist, 91)
    ret_1m  = _hist_return(hist, 30)

    score = 0
    reasons = []

    # 1. Price vs 50-DMA (short/medium trend)
    if cmp > dma50:
        score += 25; reasons.append("Above 50-DMA (uptrend)")
    else:
        score -= 25; reasons.append("Below 50-DMA (downtrend)")

    # 2. Price vs 200-DMA (long trend)
    if dma200:
        if cmp > dma200:
            score += 20; reasons.append("Above 200-DMA (long-term bullish)")
        else:
            score -= 20; reasons.append("Below 200-DMA (long-term bearish)")
        # 3. Golden / death cross
        if dma50 > dma200:
            score += 10; reasons.append("Golden cross (50>200)")
        else:
            score -= 10; reasons.append("Death cross (50<200)")

    # 4. RSI momentum
    if rsi is not None:
        if rsi >= 70:
            score -= 15; reasons.append(f"RSI {rsi} overbought")
        elif rsi >= 55:
            score += 12; reasons.append(f"RSI {rsi} bullish momentum")
        elif rsi <= 30:
            score += 10; reasons.append(f"RSI {rsi} oversold (bounce setup)")
        elif rsi <= 45:
            score -= 12; reasons.append(f"RSI {rsi} weak momentum")

    # 5. Position in 52-week range
    if hi52 > lo52:
        pos = (cmp - lo52) / (hi52 - lo52) * 100
        if pos >= 90:
            score -= 8; reasons.append(f"Near 52W high ({pos:.0f}% of range)")
        elif pos <= 15:
            score += 8; reasons.append(f"Near 52W low ({pos:.0f}% of range)")

    # 6. 3-month return trend
    if ret_3m is not None:
        if ret_3m > 8:
            score += 10; reasons.append(f"Strong 3M trend (+{ret_3m}%)")
        elif ret_3m < -8:
            score -= 10; reasons.append(f"Weak 3M trend ({ret_3m}%)")

    score = max(-100, min(100, score))

    # Map score → signal
    if score >= 35:
        signal = "BUY"
    elif score <= -35:
        signal = "SELL"
    else:
        signal = "HOLD"

    # Confidence from absolute score
    a = abs(score)
    confidence = "High" if a >= 60 else "Medium" if a >= 35 else "Low"

    # Trade levels (only meaningful for BUY)
    entry = target = stop_loss = None
    if signal == "BUY":
        atr_proxy = (hi52 - lo52) * 0.03   # rough volatility unit
        entry     = round(cmp, 2)
        stop_loss = round(min(dma50, cmp - 2 * atr_proxy), 2)
        target    = round(cmp + 3 * atr_proxy, 2)
    elif signal == "SELL":
        entry     = round(cmp, 2)   # exit level

    out.update({
        "signal": signal, "score": int(score), "confidence": confidence,
        "reasons": reasons, "rsi": rsi,
        "dma50": round(dma50, 2), "dma200": round(dma200, 2) if dma200 else None,
        "entry": entry, "target": target, "stop_loss": stop_loss,
        "pos_in_range": round((cmp - lo52) / (hi52 - lo52) * 100, 1) if hi52 > lo52 else None,
    })
    return out


def get_etf_quote(symbol):
    """Current price, day change, 52w range, trailing returns, AND signal for an ETF."""
    name = ETF_UNIVERSE.get(symbol, symbol)
    hist = get_etf_history(symbol, period="1y")
    live = _etf_fast_price(symbol)

    cmp = prev = high52 = low52 = None
    day_chg = None
    if hist is not None and not hist.empty and "Close" in hist.columns:
        closes = hist["Close"].dropna()
        if not closes.empty:
            cmp = round(live if live else float(closes.iloc[-1]), 2)
            if len(closes) >= 2:
                prev = float(closes.iloc[-2])
                day_chg = round((cmp / prev - 1) * 100, 2)
            high52 = round(float(closes.max()), 2)
            low52  = round(float(closes.min()), 2)
    elif live:
        cmp = round(live, 2)

    # Technical signal (reuses the already-fetched history — no extra network call)
    sig = compute_etf_signal(symbol, hist=hist)

    return {
        "symbol": symbol,
        "name": name,
        "cmp": cmp,
        "day_chg": day_chg,
        "high52": high52,
        "low52": low52,
        "returns": {
            "1M": _hist_return(hist, 30),
            "3M": _hist_return(hist, 91),
            "6M": _hist_return(hist, 182),
            "1Y": _hist_return(hist, 365),
        },
        "signal": sig["signal"],
        "signal_score": sig["score"],
        "signal_confidence": sig["confidence"],
        "signal_reasons": sig["reasons"],
        "rsi": sig["rsi"],
        "dma50": sig["dma50"],
        "dma200": sig["dma200"],
        "entry": sig["entry"],
        "target": sig["target"],
        "stop_loss": sig["stop_loss"],
        "has_data": cmp is not None,
    }


def etf_summary(symbol):
    """Alias for get_etf_quote — symmetry with mf_summary."""
    return get_etf_quote(symbol)


def scan_etfs(symbols=None):
    """Fetch quotes for all (or a subset of) ETFs → DataFrame sorted by 1Y return.
    Resilient: ETFs that fail to fetch are skipped, not fatal."""
    syms = symbols or list(ETF_UNIVERSE.keys())
    rows = []
    for sym in syms:
        try:
            q = get_etf_quote(sym)
            if not q.get("has_data"):
                continue
            r = q["returns"]
            rows.append({
                "Symbol": sym,
                "Name": q["name"],
                "Signal": q["signal"],
                "Score": q["signal_score"],
                "CMP": q["cmp"],
                "Day %": q["day_chg"],
                "RSI": q["rsi"],
                "1M %": r.get("1M"), "3M %": r.get("3M"),
                "6M %": r.get("6M"), "1Y %": r.get("1Y"),
                "52W High": q["high52"], "52W Low": q["low52"],
            })
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if not df.empty and "Score" in df.columns:
        df = df.sort_values("Score", ascending=False, na_position="last")
        df = df.reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
#  Self-test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("ETF universe size:", len(ETF_UNIVERSE))
    print("Sample MF search (needs network):")
    try:
        res = search_mf("hdfc small cap")
        for r in res[:5]:
            print("  ", r["schemeCode"], r["schemeName"][:50])
    except Exception as e:
        print("  (offline)", e)
