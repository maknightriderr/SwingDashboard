"""
earnings_guard.py — how close is a stock to reporting results?

Why this exists: the scanner can hand you a textbook breakout that reports
earnings in three days. That is not a setup, it is a coin flip — the chart is
irrelevant next to the number that is about to print. The system was completely
blind to this, which meant you were taking that risk without being told.

DESIGN — deliberately BOUNDED
-----------------------------
Fetching earnings dates for 2,000+ symbols is exactly the mistake that hung the
scanner when Angel LTP was called per-symbol. So this never runs over the whole
universe: it takes a SMALL list (the rows you are actually looking at), runs
threaded with a HARD deadline, and returns partial results rather than blocking.

Dates are cached for a day because an earnings date does not move intraday.
"""

import time
import datetime as _dt
import concurrent.futures as _cf

_CACHE = {}          # symbol -> (fetched_epoch, iso_date_or_None)
_CACHE_TTL = 86400   # 1 day


def _clean(symbol):
    s = str(symbol).upper().strip()
    for sfx in (".NS", ".BO", ".NSE", ".BSE"):
        if s.endswith(sfx):
            s = s[: -len(sfx)]
    return s


def _fetch_one(symbol):
    """Next earnings date for one symbol, as an ISO string, or None."""
    sym = _clean(symbol)
    now = time.time()
    hit = _CACHE.get(sym)
    if hit and (now - hit[0]) < _CACHE_TTL:
        return sym, hit[1]

    date_str = None
    try:
        import pandas as pd
        import yfinance as yf
        cal = yf.Ticker(sym + ".NS").calendar
        raw = None
        if isinstance(cal, dict):
            raw = cal.get("Earnings Date") or cal.get("earningsDate")
        elif cal is not None and hasattr(cal, "columns"):
            for col in ("Earnings Date", "earningsDate"):
                if col in cal.columns and len(cal[col]) and pd.notna(cal[col].iloc[0]):
                    raw = cal[col].iloc[0]
                    break
        if raw is not None:
            # yfinance sometimes returns a list/range of candidate dates
            if isinstance(raw, (list, tuple)) and raw:
                raw = raw[0]
            date_str = str(pd.Timestamp(raw).date())
    except Exception:
        date_str = None

    _CACHE[sym] = (now, date_str)
    return sym, date_str


def earnings_proximity(symbols, max_symbols=60, deadline_s=12):
    """{symbol: {date, days_away, window}} for a SMALL list of symbols.

    window: 'imminent' (<=3 days), 'near' (<=7), 'soon' (<=14), 'clear' (>14)
            or None when the date is unknown.

    Never raises, never blocks past the deadline, and silently returns whatever
    arrived in time — an unknown earnings date must not stall the page.
    """
    out = {}
    syms = list(dict.fromkeys([_clean(s) for s in symbols if s]))[:max_symbols]
    if not syms:
        return out

    ex = _cf.ThreadPoolExecutor(max_workers=8)
    try:
        futs = [ex.submit(_fetch_one, s) for s in syms]
        try:
            for f in _cf.as_completed(futs, timeout=deadline_s):
                sym, date_str = f.result()
                out[sym] = _describe(date_str)
        except _cf.TimeoutError:
            pass
    except Exception:
        pass
    finally:
        ex.shutdown(wait=False)
    return out


def _describe(date_str):
    if not date_str:
        return {"date": None, "days_away": None, "window": None}
    try:
        d = _dt.date.fromisoformat(date_str)
    except Exception:
        return {"date": None, "days_away": None, "window": None}
    days = (d - _dt.date.today()).days
    if days < 0:
        # Already reported — the risk has passed, not upcoming.
        window = "clear"
    elif days <= 3:
        window = "imminent"
    elif days <= 7:
        window = "near"
    elif days <= 14:
        window = "soon"
    else:
        window = "clear"
    return {"date": date_str, "days_away": days, "window": window}


def earnings_note(info):
    """One-line warning for the UI, or '' when there is nothing to say."""
    if not info or not info.get("window"):
        return ""
    d, w = info.get("days_away"), info["window"]
    if w == "imminent":
        return (f"🚨 Reports in {d} day(s) ({info['date']}) — a breakout entry "
                f"here is a bet on the number, not the chart")
    if w == "near":
        return (f"⚠️ Reports in {d} days ({info['date']}) — size down or wait "
                f"for the print")
    if w == "soon":
        return f"📅 Reports in {d} days ({info['date']})"
    return ""


def earnings_penalty(info):
    """Score adjustment. Deliberately small: earnings proximity is a RISK flag,
    not a verdict on the setup. It should nudge ranking and warn you, not
    silently delete otherwise-good candidates."""
    if not info or not info.get("window"):
        return 0
    return {"imminent": -3, "near": -2, "soon": 0, "clear": 0}.get(info["window"], 0)
