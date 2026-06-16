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


def mf_summary(scheme_code):
    """Everything needed to render a fund card: NAV, meta, returns, 52w range."""
    nav = get_mf_nav(scheme_code)
    hist = get_mf_history(scheme_code)
    returns = get_mf_returns(scheme_code)

    high52 = low52 = None
    if hist is not None and not hist.empty:
        last_year = hist[hist["date"] >= (hist["date"].iloc[-1] - timedelta(days=365))]
        if not last_year.empty:
            high52 = round(float(last_year["nav"].max()), 2)
            low52  = round(float(last_year["nav"].min()), 2)

    return {
        **nav,
        "returns": returns,
        "high52": high52,
        "low52": low52,
        "has_history": hist is not None and not hist.empty,
    }


def compare_mfs(scheme_codes):
    """Side-by-side comparison table for multiple funds.
    Returns a DataFrame with one row per fund."""
    rows = []
    for code in scheme_codes:
        s = mf_summary(code)
        r = s.get("returns", {})
        rows.append({
            "Fund": s.get("scheme_name", str(code))[:45],
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


def get_etf_quote(symbol):
    """Current price, day change, 52w range, and trailing returns for an ETF."""
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
                "CMP": q["cmp"],
                "Day %": q["day_chg"],
                "1M %": r.get("1M"), "3M %": r.get("3M"),
                "6M %": r.get("6M"), "1Y %": r.get("1Y"),
                "52W High": q["high52"], "52W Low": q["low52"],
            })
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if not df.empty and "1Y %" in df.columns:
        df = df.sort_values("1Y %", ascending=False, na_position="last")
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
