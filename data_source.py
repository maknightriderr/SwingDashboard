"""
data_source.py — Unified market-data layer: Angel One PRIMARY, Yahoo FALLBACK.

Design goal: your scanners keep calling ONE function and don't care where the
data came from. Angel One is tried first (delay-free, your own broker feed);
if Angel is unavailable for any reason — not logged in, rate-limited past
retries, symbol not in Angel's map, missing secrets, or any exception — this
transparently falls back to Yahoo so a scan never dies just because Angel had
a bad moment.

Why a separate layer (instead of editing signals.py's fetchers directly):
  - signals.py fetches history in ~7 places (each scanner + regime + sector).
    Editing all of them is the exact kind of wide change that caused the
    _q / RS-API / VCP-key regressions earlier. Instead, signals.py changes in
    ONE line (see WIRING below) to route every fetch through here.
  - Fallback logic lives in exactly one place, so it's testable and can't
    drift between scanners.

WIRING (one change in signals.py — do this after the pilot is stable):
  At the TOP of signals.py, right after `import yfinance as yf`, add:

      try:
          import data_source as _ds
          _USE_UNIFIED = True
      except Exception:
          _USE_UNIFIED = False

  Then find signals.py's `def _bulk_fetch_history(symbols, period="1y"):`
  and make its FIRST lines:

      def _bulk_fetch_history(symbols, period="1y"):
          if _USE_UNIFIED:
              return _ds.bulk_fetch_history(symbols, period=period)
          # ... existing Yahoo body stays below as the ultimate fallback ...

  That single guard routes every scanner (Universe, Trap, SMC, VCP, RS,
  Sector) through Angel-primary/Yahoo-fallback without touching them.

SECRETS (same as the pilot — Streamlit secrets or env vars):
    angel_api_key, angel_client_code, angel_pin, angel_totp_secret
  If any are missing, this module silently runs Yahoo-only (no crash) — so it's
  safe to deploy before Angel is fully configured.
"""

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Angel pilot module (from the validated pilot). If it can't import for any
# reason, we degrade to Yahoo-only rather than breaking the whole app.
try:
    import angel_data as _angel
    _ANGEL_AVAILABLE = True
except Exception:
    _angel = None
    _ANGEL_AVAILABLE = False

import yfinance as yf


# ── Period string → number of calendar days (Angel wants a day count) ─────────
_PERIOD_DAYS = {
    "1mo": 30, "3mo": 92, "6mo": 183, "1y": 366, "2y": 731,
    "ytd": 250, "max": 2000,
}


def _period_to_days(period):
    return _PERIOD_DAYS.get(str(period).lower(), 183)


# Angel fetches ONE symbol at a time at ~2 req/sec. That's fine for a watchlist,
# but a full universe (2000+ symbols) would take ~20 minutes sequentially —
# far longer than Streamlit Cloud allows a single script run, so the scan gets
# killed mid-way and never returns. For any list bigger than this, we skip Angel
# and use fast THREADED Yahoo instead. Angel stays the primary source for small
# lists (live prices, single-stock lookups, watchlist) where delay-free matters.
_ANGEL_BULK_MAX = 50


# ── Runtime health flag: once Angel login is confirmed dead for this session,
#    stop hammering it and go straight to Yahoo (re-checked each new day). ─────
_angel_state = {"checked_date": None, "healthy": False}


def _angel_is_healthy():
    """Cheap gate: only attempt Angel if secrets exist AND login works today.
    Cached per calendar day so we don't re-probe login on every symbol."""
    if not _ANGEL_AVAILABLE:
        return False
    import datetime as _dt
    today = _dt.datetime.now().date()
    if _angel_state["checked_date"] == today:
        return _angel_state["healthy"]
    # Probe once per day
    ok = False
    try:
        conn = _angel.get_connection()
        ok = conn is not None
    except Exception:
        ok = False
    _angel_state["checked_date"] = today
    _angel_state["healthy"] = ok
    if ok:
        print("[data_source] Angel One is primary today ✅")
    else:
        print("[data_source] Angel One unavailable — using Yahoo fallback for today")
    return ok


# ── Yahoo single-symbol fetch (mirrors signals.py's normalisation) ────────────
def _yahoo_bulk_threaded(symbols, period="6mo"):
    """Fast parallel Yahoo fetch for large universes (5 workers). Returns
    {symbol: DataFrame}. Index symbols route through fetch_index."""
    results = {}
    def _one(sym):
        if str(sym).startswith("^"):
            return sym, fetch_index(sym, period)
        return sym, _yahoo_fetch(sym, period)
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(_one, s): s for s in symbols}
        for f in as_completed(futs):
            try:
                sym, df = f.result()
                if df is not None and not df.empty:
                    results[sym] = df
            except Exception:
                continue
    return results


def _is_market_closed_today_ist():
    """True if it's a weekday AFTER NSE close (15:30 IST) — the window where
    today's completed daily bar SHOULD exist. Weekends/holidays return False
    (yesterday's bar is then correctly the latest)."""
    import datetime as _dt
    now_ist = _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=5, minutes=30)))
    if now_ist.weekday() >= 5:          # Sat/Sun
        return False
    return (now_ist.hour, now_ist.minute) >= (15, 45)   # 15-min buffer after close


def _topup_today_bar(df, symbol):
    """Yahoo's daily-history endpoint sometimes lags after the close — the
    session has ended but today's bar is missing, so every indicator runs one
    day behind (user saw 'Data Date = yesterday' after a post-close scan).
    When that happens (weekday, post-close, last bar < today), fetch today's
    OHLCV from the live-quote endpoint (fast_info — a different, fresher
    endpoint) and append it as today's completed bar."""
    try:
        import datetime as _dt
        if df is None or df.empty or not _is_market_closed_today_ist():
            return df
        today = _dt.datetime.now(
            _dt.timezone(_dt.timedelta(hours=5, minutes=30))).date()
        last = df.index[-1].date() if hasattr(df.index[-1], "date") else None
        if last is None or last >= today:
            return df                      # already current
        clean = str(symbol).upper().replace(".NS", "").replace(".BO", "")
        fi = yf.Ticker(clean + ".NS").fast_info
        o = fi.get("open");      h = fi.get("dayHigh")
        l = fi.get("dayLow");    c = fi.get("lastPrice") or fi.get("last_price")
        v = fi.get("lastVolume") or fi.get("last_volume") or 0
        if not c or not o:
            return df                      # quote unavailable — keep what we have
        row = pd.DataFrame(
            {"Open": [float(o)], "High": [float(h or max(o, c))],
             "Low": [float(l or min(o, c))], "Close": [float(c)],
             "Volume": [float(v)]},
            index=pd.DatetimeIndex([pd.Timestamp(today)]))
        return pd.concat([df, row])
    except Exception:
        return df                          # never break a fetch over a top-up


def _yahoo_fetch(symbol, period="6mo"):
    clean = str(symbol).upper().replace(".NS", "").replace(".BO", "")
    for suffix in (".NS", ".BO"):
        try:
            df = yf.Ticker(clean + suffix).history(
                period=period, interval="1d", auto_adjust=False)
            if df is not None and not df.empty and "Close" in df.columns:
                out = pd.DataFrame({
                    "Open":   df["Open"]   if "Open"   in df else df["Close"],
                    "High":   df["High"]   if "High"   in df else df["Close"],
                    "Low":    df["Low"]    if "Low"    in df else df["Close"],
                    "Close":  df["Close"],
                    "Volume": df["Volume"] if "Volume" in df else 0,
                }).dropna(subset=["Close"]).ffill().bfill()
                if not out.empty:
                    return _topup_today_bar(out, symbol)
        except Exception:
            continue
    return None


# ── Index fetch (Nifty / sector indices) — Angel has different tokens for
#    indices, and the pilot only mapped equities, so indices ALWAYS use Yahoo
#    for now. This keeps Market Regime / Sector Rotation correct. ─────────────
def fetch_index(symbol, period="1y"):
    """Index symbols (^NSEI, ^CNXIT, ...) → Yahoo (Angel index tokens are a
    later phase). Returns a normalised OHLCV DataFrame or None."""
    try:
        df = yf.Ticker(symbol).history(period=period, interval="1d",
                                       auto_adjust=False)
        if df is not None and not df.empty and "Close" in df.columns:
            return df.dropna(subset=["Close"]).ffill().bfill()
    except Exception:
        pass
    return None


# ── THE unified single-symbol fetch: Angel first, Yahoo on any failure ───────
def fetch_history(symbol, period="6mo"):
    """One equity symbol → normalised OHLCV DataFrame (Angel primary, Yahoo
    fallback), or None if BOTH sources fail. Index symbols (starting '^')
    always route to Yahoo."""
    if str(symbol).startswith("^"):
        return fetch_index(symbol, period)

    days = _period_to_days(period)

    # 1) Try Angel (only if healthy today)
    if _angel_is_healthy():
        try:
            df = _angel.fetch_history(symbol, days=days, interval="ONE_DAY")
            if df is not None and not df.empty and "Close" in df.columns:
                return df
        except Exception:
            pass   # fall through to Yahoo

    # 2) Fallback: Yahoo
    return _yahoo_fetch(symbol, period)


# ── Bulk fetch — the function signals.py actually calls. Preserves Yahoo's
#    return shape exactly: {symbol: DataFrame} (missing symbols simply absent).
#    Angel is sequential (rate-limited); Yahoo fallback is per-symbol so one
#    bad symbol never sinks the batch. ─────────────────────────────────────────
def bulk_fetch_history(symbols, period="1y"):
    """{symbol: DataFrame} for a list of symbols.

    Large lists (full universe) → fast THREADED Yahoo, skipping Angel entirely
    (sequential Angel would take ~20 min and get killed by Streamlit). Small
    lists → Angel-primary per symbol with per-symbol Yahoo fallback.
    Index symbols route to Yahoo automatically either way."""
    symbols = list(symbols)
    # Big universe → fast parallel Yahoo (Angel is too slow sequentially here)
    if len(symbols) > _ANGEL_BULK_MAX:
        print(f"[data_source] {len(symbols)} symbols > {_ANGEL_BULK_MAX} — using "
              f"fast threaded Yahoo for this scan (Angel reserved for live "
              f"prices / watchlist / single-stock).")
        return _yahoo_bulk_threaded(symbols, period)

    results = {}
    angel_ok = _angel_is_healthy()
    yahoo_fallback_count = 0

    for sym in symbols:
        if str(sym).startswith("^"):
            df = fetch_index(sym, period)
            if df is not None:
                results[sym] = df
            continue

        df = None
        if angel_ok:
            try:
                df = _angel.fetch_history(sym, days=_period_to_days(period),
                                          interval="ONE_DAY")
            except Exception:
                df = None
        if df is None or df.empty:
            df = _yahoo_fetch(sym, period)
            if df is not None:
                yahoo_fallback_count += 1
        if df is not None and not df.empty:
            results[sym] = df

    if angel_ok and yahoo_fallback_count:
        print(f"[data_source] {yahoo_fallback_count}/{len(symbols)} symbols fell "
              f"back to Yahoo (Angel missing/failed for those)")
    return results


# ── Live price: Angel LTP (delay-free) with Yahoo fast_info fallback ──────────
def get_live_price(symbol):
    """Real-time price for one equity symbol. Angel LTP first (genuinely
    delay-free), Yahoo fast_info as fallback. Returns float or None."""
    if _angel_is_healthy():
        try:
            ltp = _angel.get_ltp(symbol)
            if ltp:
                return float(ltp)
        except Exception:
            pass
    try:
        clean = str(symbol).upper().replace(".NS", "").replace(".BO", "")
        fi = yf.Ticker(clean + ".NS").fast_info
        px = fi.get("lastPrice") or fi.get("last_price")
        return float(px) if px else None
    except Exception:
        return None


def source_status():
    """Small helper for a dashboard badge: which source is active right now."""
    if _angel_is_healthy():
        return {"primary": "Angel One", "status": "live", "fallback": "Yahoo"}
    return {"primary": "Yahoo", "status": "fallback",
            "fallback": "none (Angel unavailable)"}


# ── Self-test (needs Angel secrets set as env vars for the Angel path; runs
#    Yahoo-only cleanly without them) ───────────────────────────────────────────
if __name__ == "__main__":
    WATCH = ["RELIANCE", "TCS", "HDFCBANK", "^NSEI"]
    print("=== source status ===")
    print(" ", source_status())
    print("\n=== bulk history (mixed equities + index) ===")
    data = bulk_fetch_history(WATCH, period="6mo")
    for s, df in data.items():
        print(f"  {s}: {len(df)} bars, latest close ₹{df['Close'].iloc[-1]:.2f}")
    print("\n=== live prices ===")
    for s in ["RELIANCE", "TCS", "HDFCBANK"]:
        print(f"  {s}: ₹{get_live_price(s)}")
