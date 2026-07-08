"""
angel_data.py — PILOT: Angel One SmartAPI as a delay-free data source.

Scope (deliberately small): a HANDFUL of symbols — your open positions and/or
watchlist — NOT the full 2,300-symbol universe. The point of this pilot is to
prove the auth flow is reliable on Streamlit Cloud before any wider rollout.

Why this exists (recap):
  - Yahoo daily bars carry a real intraday delay + occasional silent rate-limit
    blocks. Angel One SmartAPI is your OWN broker's feed — LTP (get_ltp) is
    genuinely real-time, and historical candles lag only ~5s intraday, not
    15-20 min.
  - The real cost isn't money (SmartAPI is free) — it's a DAILY login (session
    expires at midnight IST) and a hard 3 requests/second limit on candle data.
    This module handles both so the rest of your app doesn't have to think
    about them.

Drop-in contract: fetch_history() / bulk_fetch_history() return the SAME
shape as signals.py's Yahoo fetchers — a DataFrame with Open/High/Low/Close/
Volume columns and a DatetimeIndex — so scanner code doesn't need to change
to test this, you just point it at a different fetcher for a pilot symbol list.

SETUP REQUIRED (do this before running):
  1. pip install smartapi-python pyotp logzero websocket-client
     (add these 4 lines to requirements.txt)
  2. In Streamlit secrets (or a local .env for testing), set:
        angel_api_key       = "<your SmartAPI app's API key>"
        angel_client_code   = "S1278713"     # your Angel One client ID
        angel_pin           = "<your 4-digit trading PIN>"
        angel_totp_secret   = "<the TOTP secret shown when you set up 2FA>"
     The TOTP secret is the base32 string you scanned as a QR code when you
     first enabled the authenticator app on your Angel One account — NOT the
     6-digit code itself (that changes every 30s; this module generates it
     fresh on every login using pyotp).
  3. Test with a SMALL symbol list first (see the __main__ block at the
     bottom of this file) before wiring it into any scanner.

NOT YET DONE (by design, for a later phase once the pilot proves stable):
  - Full-universe (2,300 symbol) fetching — at 3 req/sec that's ~13 minutes
    serialized; fine for a nightly job, not yet wired into live scanning.
  - Automatic fallback to Yahoo if Angel's session/login fails.
  - Persisting the scrip-master token map in Postgres instead of /tmp (which
    Streamlit Cloud can wipe between container restarts).
"""

import os
import json
import time
import threading
import requests
import pandas as pd
from datetime import datetime, timedelta

try:
    import pyotp
except ImportError:
    pyotp = None

try:
    from SmartApi import SmartConnect
except ImportError:
    SmartConnect = None


# ══════════════════════════════════════════════════════════════════════════
# Config — reads from st.secrets when running inside Streamlit, else env vars
# (so this module also works in a plain python test script, e.g. Codespaces)
# ══════════════════════════════════════════════════════════════════════════
def _get_secret(key, default=None):
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


API_KEY     = _get_secret("angel_api_key")
CLIENT_CODE = _get_secret("angel_client_code")
PIN         = _get_secret("angel_pin")
TOTP_SECRET = _get_secret("angel_totp_secret")

_SCRIP_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
_SCRIP_CACHE_PATH = "/tmp/angel_scrip_master.json"
_SCRIP_CACHE_TTL  = 24 * 3600  # refresh once a day — instrument list rarely changes intraday

_RATE_LIMIT_PER_SEC = 2   # SmartAPI documents 3 req/sec for getCandleData,
                          # but real-world enforcement is flaky — several
                          # users report 403s well under the documented
                          # limit. 2/sec + a bigger buffer (below) trades a
                          # little speed for reliability.
_rate_lock = threading.Lock()
_call_timestamps = []

_session = {"conn": None, "login_date": None, "jwt": None,
           "feed": None, "refresh": None}
_last_failure = {"ts": 0, "message": None}
_FAILURE_COOLDOWN_SEC = 120   # don't retry a failed login for 2 min —
                              # protects Angel's 5-attempt MPIN lockout
                              # from being burned by repeated calls


# ══════════════════════════════════════════════════════════════════════════
# Throttle — never exceed 3 req/sec on getCandleData, however many symbols
# bulk_fetch_history is asked for. Blocks (sleeps), never drops a request.
# ══════════════════════════════════════════════════════════════════════════
def _throttle():
    with _rate_lock:
        now = time.time()
        while _call_timestamps and now - _call_timestamps[0] > 1.0:
            _call_timestamps.pop(0)
        if len(_call_timestamps) >= _RATE_LIMIT_PER_SEC:
            sleep_for = 1.0 - (now - _call_timestamps[0]) + 0.25   # bigger safety buffer
            if sleep_for > 0:
                time.sleep(sleep_for)
        _call_timestamps.append(time.time())


# ══════════════════════════════════════════════════════════════════════════
# Daily TOTP login — SmartAPI sessions expire at midnight IST regardless of
# activity, so we re-login once per calendar day and cache the connection.
# ══════════════════════════════════════════════════════════════════════════
def _login():
    import time as _t
    if (_last_failure["ts"] and
            _t.time() - _last_failure["ts"] < _FAILURE_COOLDOWN_SEC):
        print(f"[angel_data] skipping login retry — last attempt failed "
              f"{int(_t.time()-_last_failure['ts'])}s ago "
              f"({_last_failure['message']}). Waiting out the "
              f"{_FAILURE_COOLDOWN_SEC}s cooldown to protect your "
              f"5-attempt MPIN quota. Fix the credential, then wait "
              f"or restart.")
        return None
    if SmartConnect is None or pyotp is None:
        print("[angel_data] smartapi-python or pyotp not installed — "
              "run: pip install smartapi-python pyotp logzero websocket-client")
        return None
    missing = [n for n, v in [("angel_api_key", API_KEY),
                              ("angel_client_code", CLIENT_CODE),
                              ("angel_pin", PIN),
                              ("angel_totp_secret", TOTP_SECRET)] if not v]
    if missing:
        print(f"[angel_data] missing secrets: {missing}")
        return None
    try:
        obj = SmartConnect(api_key=API_KEY)
        totp_code = pyotp.TOTP(TOTP_SECRET).now()
        data = obj.generateSession(CLIENT_CODE, PIN, totp_code)
        if not data or not data.get("status"):
            _last_failure["ts"] = __import__("time").time()
            _last_failure["message"] = (data or {}).get("message", "unknown")
            print(f"[angel_data] login rejected: {data}")
            print(f"[angel_data] ⚠️ cooling down {_FAILURE_COOLDOWN_SEC}s before any "
                  f"further attempt — this run will NOT retry login again "
                  f"regardless of how many fetch calls follow.")
            return None
        _session["conn"]       = obj
        _session["login_date"] = datetime.now().date()
        _session["jwt"]        = data["data"]["jwtToken"]
        _session["feed"]       = data["data"]["feedToken"]
        _session["refresh"]    = data["data"]["refreshToken"]
        print(f"[angel_data] logged in fresh for {_session['login_date']}")
        return obj
    except Exception as e:
        _last_failure["ts"] = __import__("time").time()
        _last_failure["message"] = str(e)
        print(f"[angel_data] login exception: {e}")
        print(f"[angel_data] ⚠️ cooling down {_FAILURE_COOLDOWN_SEC}s before any "
              f"further attempt.")
        return None


def get_connection():
    """Live SmartConnect session — re-logs in automatically if the day
    rolled over or no session exists yet. Cheap to call on every use."""
    today = datetime.now().date()
    if _session["conn"] is not None and _session["login_date"] == today:
        return _session["conn"]
    return _login()


def force_relogin():
    """Call this if a request comes back with an auth/expired-token error
    mid-session — clears the cache so the next get_connection() re-logs in."""
    _session["conn"] = None
    _session["login_date"] = None


# ══════════════════════════════════════════════════════════════════════════
# Scrip master — maps plain NSE symbols ("RELIANCE") to Angel's symboltoken.
# Cached to /tmp for 24h (Streamlit Cloud can wipe /tmp on restart — that's
# fine, it just re-downloads once, ~2-3MB file).
# ══════════════════════════════════════════════════════════════════════════
_scrip_map = {}


def _load_scrip_master():
    if _scrip_map:
        return _scrip_map
    if os.path.exists(_SCRIP_CACHE_PATH):
        age = time.time() - os.path.getmtime(_SCRIP_CACHE_PATH)
        if age < _SCRIP_CACHE_TTL:
            try:
                with open(_SCRIP_CACHE_PATH) as f:
                    raw = json.load(f)
                _build_scrip_map(raw)
                return _scrip_map
            except Exception:
                pass
    try:
        resp = requests.get(_SCRIP_MASTER_URL, timeout=30)
        raw = resp.json()
        try:
            with open(_SCRIP_CACHE_PATH, "w") as f:
                json.dump(raw, f)
        except Exception:
            pass   # cache write failing is fine, we still have `raw` in memory
        _build_scrip_map(raw)
    except Exception as e:
        print(f"[angel_data] scrip master download failed: {e}")
    return _scrip_map


def _build_scrip_map(raw):
    """Keep only NSE cash-market equities (the '-EQ' series), matching the
    symbol convention your dashboard already uses (no .NS/.BO suffix)."""
    global _scrip_map
    m = {}
    for row in raw:
        try:
            if row.get("exch_seg") != "NSE":
                continue
            sym = row.get("symbol", "")
            if not sym.endswith("-EQ"):
                continue
            clean = sym[:-3]   # strip "-EQ"
            m[clean] = row.get("token")
        except Exception:
            continue
    _scrip_map = m


def get_token(symbol):
    """NSE symbol → Angel symboltoken, or None if not found."""
    m = _load_scrip_master()
    return m.get(str(symbol).upper().strip())


# ══════════════════════════════════════════════════════════════════════════
# Historical candles — SAME return shape as signals.py's Yahoo fetchers, so
# this is a drop-in for a pilot symbol list (open positions / watchlist).
# ══════════════════════════════════════════════════════════════════════════
_INTERVAL_MAX_DAYS = {
    "ONE_MINUTE": 30, "THREE_MINUTE": 60, "FIVE_MINUTE": 100,
    "TEN_MINUTE": 100, "FIFTEEN_MINUTE": 200, "THIRTY_MINUTE": 200,
    "ONE_HOUR": 400, "ONE_DAY": 2000,
}


def fetch_history(symbol, days=180, interval="ONE_DAY", _retry=True, _rate_retries=2):
    """Fetch OHLCV history for ONE NSE equity symbol.
    Returns a DataFrame(Open, High, Low, Close, Volume) indexed by datetime,
    or None on any failure — mirrors signals.py's _fetch_history contract."""
    conn = get_connection()
    if conn is None:
        return None
    token = get_token(symbol)
    if not token:
        print(f"[angel_data] no symboltoken found for {symbol} "
              f"(check spelling / it may not be an NSE cash-market symbol)")
        return None

    max_days = _INTERVAL_MAX_DAYS.get(interval, 2000)
    days = min(days, max_days)
    todate = datetime.now()
    fromdate = todate - timedelta(days=days)
    params = {
        "exchange": "NSE",
        "symboltoken": str(token),
        "interval": interval,
        "fromdate": fromdate.strftime("%Y-%m-%d %H:%M"),
        "todate": todate.strftime("%Y-%m-%d %H:%M"),
    }

    _throttle()
    try:
        resp = conn.getCandleData(params)
    except Exception as e:
        msg = str(e)
        # "Access denied because of exceeding access rate" is a TRANSIENT
        # rate-limit hit, not a credential problem — back off and retry a
        # couple of times before giving up on this symbol.
        if "exceeding access rate" in msg.lower() and _rate_retries > 0:
            backoff = 1.5 if _rate_retries == 2 else 3.0
            print(f"[angel_data] rate-limited on {symbol}, backing off {backoff}s "
                  f"({_rate_retries} retries left)")
            time.sleep(backoff)
            return fetch_history(symbol, days=days, interval=interval,
                                 _retry=_retry, _rate_retries=_rate_retries - 1)
        print(f"[angel_data] getCandleData exception for {symbol}: {e}")
        return None

    if not resp or not resp.get("status"):
        msg = (resp or {}).get("message", "")
        # Session likely expired mid-day — re-login once and retry.
        if _retry and ("token" in msg.lower() or "session" in msg.lower()
                       or "auth" in msg.lower()):
            force_relogin()
            return fetch_history(symbol, days=days, interval=interval, _retry=False)
        print(f"[angel_data] getCandleData failed for {symbol}: {msg}")
        return None

    rows = resp.get("data") or []
    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)
    return df


def bulk_fetch_history(symbols, days=180, interval="ONE_DAY"):
    """Sequential fetch across a symbol list, respecting the 3 req/sec
    throttle internally. PILOT SCOPE: use this for a small list (watchlist /
    open positions) — NOT the full universe (would take ~13 min at 2,300
    symbols, and this pilot hasn't been load-tested at that scale yet).
    Returns {symbol: DataFrame}."""
    results = {}
    for sym in symbols:
        df = fetch_history(sym, days=days, interval=interval)
        if df is not None:
            results[sym] = df
    return results


# ══════════════════════════════════════════════════════════════════════════
# Real-time LTP — this is the genuinely delay-free piece (broker's own tick,
# not a daily bar). Use this for "live price" displays; use fetch_history()
# for the OHLCV series scanners need.
# ══════════════════════════════════════════════════════════════════════════
def get_ltp(symbol, _retry=True):
    """Real-time last-traded-price for one NSE equity symbol, or None."""
    conn = get_connection()
    if conn is None:
        return None
    token = get_token(symbol)
    if not token:
        return None
    try:
        resp = conn.ltpData("NSE", f"{symbol.upper()}-EQ", str(token))
        if resp and resp.get("status"):
            return float(resp["data"]["ltp"])
        msg = (resp or {}).get("message", "")
        if _retry and ("token" in msg.lower() or "session" in msg.lower()):
            force_relogin()
            return get_ltp(symbol, _retry=False)
    except Exception as e:
        print(f"[angel_data] ltpData exception for {symbol}: {e}")
    return None


def get_ltp_bulk(symbols):
    """Convenience wrapper: {symbol: ltp_or_None} for a small list."""
    return {s: get_ltp(s) for s in symbols}


# ══════════════════════════════════════════════════════════════════════════
# Pilot self-test — run directly: `python angel_data.py` (needs secrets set
# as env vars if not running inside Streamlit — see SETUP REQUIRED above).
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    WATCHLIST = ["RELIANCE", "TCS", "HDFCBANK"]   # keep this SMALL for the pilot

    print("=== 1. Login ===")
    conn = get_connection()
    print("✅ connected" if conn else "❌ login failed — check secrets/PIN/TOTP")

    print("\n=== 2. Scrip master (symbol → token) ===")
    for s in WATCHLIST:
        print(f"  {s}: token={get_token(s)}")

    print("\n=== 3. Historical (6 months daily) ===")
    hist = bulk_fetch_history(WATCHLIST, days=180)
    for s, df in hist.items():
        print(f"  {s}: {len(df)} bars, latest close ₹{df['Close'].iloc[-1]:.2f} "
              f"@ {df.index[-1]}")

    print("\n=== 4. Live LTP (genuinely delay-free) ===")
    for s, ltp in get_ltp_bulk(WATCHLIST).items():
        print(f"  {s}: ₹{ltp}")
