"""
Swing Trading Portfolio Dashboard v14
Fixes vs v13:
  - Premium theme pack: 6 institutional themes (was 3)
  - theme_css upgraded: animated title underline, card shimmer/lift, live-pulse
    badge, focus-glow inputs, P&L row rails, tabular numerals
  - Tab 9 scorecard updated to signals.py v12 (avg 8.4, every component >= 8)
  - signals.py v12 already deployed: unified risk engine, Wilder ATR/RSI,
    numpy Supertrend, 20-day VWAP, swing-peak Fibonacci, MACD histogram
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import sqlite3
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import hashlib

from signals import (
    generate_signals, sector_rotation, predict_sector_outlook,
    find_sector_picks, send_telegram, build_telegram_message,
    get_sector, get_market_regime, generate_market_scanner,
    SECTOR_MAP, _bulk_fetch_history, compute_indicators,
    fetch_portfolio_news, UNIVERSE_SOURCES, UNIVERSE_TOTAL,
    debug_universe_load
)

# New functions added in signals.py v12+ — imported separately so the app
# degrades gracefully if an older signals.py is deployed.
try:
    from signals import scan_for_traps as _scan_for_traps
    scan_for_traps = _scan_for_traps
except ImportError:
    scan_for_traps = None

try:
    from signals import scan_for_smc_setups as _scan_for_smc_setups
    scan_for_smc_setups = _scan_for_smc_setups
except ImportError:
    scan_for_smc_setups = None

try:
    from signals import scan_for_vcp as _scan_for_vcp
    scan_for_vcp = _scan_for_vcp
except ImportError:
    scan_for_vcp = None

try:
    from signals import scan_relative_strength as _scan_rs
    scan_relative_strength = _scan_rs
except ImportError:
    scan_relative_strength = None

try:
    from signals import (
        fetch_corporate_actions,
        fetch_bulk_corporate_actions,
        scan_corporate_actions_universe,
    )
except ImportError:
    fetch_corporate_actions        = None
    fetch_bulk_corporate_actions   = None
    scan_corporate_actions_universe = None

_TRAP_SCANNER_AVAILABLE = scan_for_traps is not None
_CORP_ACTIONS_AVAILABLE = fetch_corporate_actions is not None
_SMC_SCANNER_AVAILABLE  = scan_for_smc_setups is not None

# ── Mutual Fund & ETF module (fully separate from stock/signals logic) ─────────
try:
    import funds as _funds
    _FUNDS_AVAILABLE = True
except Exception:
    _funds = None
    _FUNDS_AVAILABLE = False

# ── Market Theme Scanner module (curated NSE theme baskets) ────────────────────
try:
    import theme_scanner as _themes
    _THEMES_AVAILABLE = True
except Exception:
    _themes = None
    _THEMES_AVAILABLE = False

# ── Performance: @st.cache_data wrappers ──────────────────────────────────────
# market_regime is global (same for all users) — safe to cache across sessions.
# TTL 600s = 10 min. This renders the header banner in <100ms on reruns.
@st.cache_data(ttl=600, show_spinner=False)
def _scan_age_warning(timestamp_str):
    """(is_stale, human_age) for a scan timestamp like '08 Jul 2026 19:42'.
    Scans are cached in session_state and never expire on their own — so a tab
    left open for days would keep showing old prices with no indication. Older
    than 6 hours (one trading session) counts as stale.

    NOTE: scan timestamps are generated in IST (signals.py uses _now_ist()),
    but Streamlit Cloud servers run in UTC — so this must compare against IST
    "now" too, or the age would be off by 5:30 (negative, even)."""
    if not timestamp_str:
        return False, None
    try:
        _ts = datetime.strptime(str(timestamp_str), "%d %b %Y %H:%M")
    except Exception:
        return False, None
    from datetime import timezone, timedelta
    _ist_now = datetime.now(timezone(timedelta(hours=5, minutes=30))).replace(tzinfo=None)
    _age = _ist_now - _ts
    _hours = _age.total_seconds() / 3600
    if _hours < 1:
        _human = f"{int(_age.total_seconds() // 60)} min ago"
    elif _hours < 24:
        _human = f"{int(_hours)} hr ago"
    else:
        _human = f"{_age.days} day{'s' if _age.days != 1 else ''} ago"
    return _hours >= 6, _human


def _stale_banner(timestamp_str, rescan_label):
    """Show a prominent warning if a cached scan is stale (prices will be old)."""
    _stale, _age = _scan_age_warning(timestamp_str)
    if _stale:
        st.warning(
            f"⚠️ These results are from **{_age}** — prices and levels shown are "
            f"from that scan, not today. Click **{rescan_label}** to refresh.",
            icon="🕒")
    return _stale


try:
    import market_news as _mnews
    _MNEWS_AVAILABLE = True
except Exception:
    _mnews = None
    _MNEWS_AVAILABLE = False

try:
    import news_analysis as _nana
    _NANA_AVAILABLE = True
except Exception:
    _nana = None
    _NANA_AVAILABLE = False


def _cached_market_regime():
    return get_market_regime()


def _get_market_regime_safe():
    """Wrapper that avoids caching an empty (failed) regime result.
    If indices came back empty, clear the cache so the next rerun retries."""
    m = _cached_market_regime()
    if not m or not m.get("indices"):
        # Empty/failed — drop the cached empty so next call re-fetches fresh
        try:
            _cached_market_regime.clear()
        except Exception:
            pass
        # Try one direct (uncached) fetch right now
        try:
            m2 = get_market_regime()
            if m2 and m2.get("indices"):
                return m2
        except Exception:
            pass
    return m or {"regime": "Unknown", "indices": {}, "confidence": "—"}

# Price cache: 5-min TTL so KPI cards don't block on every sidebar interaction.
@st.cache_data(ttl=120, show_spinner=False)
def _cached_prices(symbols_tuple):
    """Fetch ACCURATE live prices for a tuple of symbols.

    Accuracy notes:
      • Primary source is fast_info.last_price — the real-time last traded price.
      • History fallback uses auto_adjust=FALSE so we get the ACTUAL close, not a
        dividend/split back-adjusted value (auto_adjust=True was making CMP wrong
        for stocks that recently went ex-dividend or split).
      • 2-min cache (was 5) so prices are fresher during market hours.
    Never raises — missing prices just stay absent."""
    import yfinance as _yf
    import pandas as _pd
    prices = {}
    if not symbols_tuple:
        return prices

    sym_map = {}   # ticker_ns → original sym
    for sym in symbols_tuple:
        clean = str(sym).upper().strip()
        for sfx in [".NS", ".BO", ".NSE", ".BSE"]:
            if clean.endswith(sfx):
                clean = clean[:-len(sfx)]
        sym_map[clean + ".NS"] = sym

    def _extract_fast_price(t):
        """Real-time last traded price from fast_info (most accurate source)."""
        try:
            fi = t.fast_info
        except Exception:
            return None
        for key in ("last_price", "lastPrice", "regularMarketPrice"):
            for getter in (lambda: fi.get(key) if hasattr(fi, "get") else None,
                           lambda: getattr(fi, key, None),
                           lambda: fi[key] if hasattr(fi, "__getitem__") else None):
                try:
                    v = getter()
                    if v is not None and not _pd.isna(v) and float(v) > 0:
                        return float(v)
                except Exception:
                    pass
        return None

    # ── METHOD 1: per-ticker fast_info in PARALLEL (live last traded price) ────
    # Same accuracy as before (fast_info stays the primary, real-time source);
    # only the fetching is now concurrent instead of one-by-one.
    from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _asc

    def _live_price(tk):
        base = tk[:-3]   # strip .NS
        for sfx in [".NS", ".BO"]:
            try:
                v = _extract_fast_price(_yf.Ticker(base + sfx))
                if v is not None and v > 0:
                    return round(v, 2)
            except Exception:
                continue
        return None

    if sym_map:
        with _TPE(max_workers=8) as _ex:
            _futs = {_ex.submit(_live_price, tk): sym for tk, sym in sym_map.items()}
            for _fut in _asc(_futs):
                try:
                    _v = _fut.result()
                    if _v is not None:
                        prices[_futs[_fut]] = _v
                except Exception:
                    pass

    # ── METHOD 2: batch download for any the live method missed ────────────────
    # auto_adjust=False → ACTUAL close price (not back-adjusted for div/splits)
    missing = [tk for tk, sym in sym_map.items() if sym not in prices]
    if missing:
        try:
            data = _yf.download(missing, period="5d", interval="1d",
                                auto_adjust=False, progress=False,
                                group_by="ticker", threads=True)
            if data is not None and not data.empty:
                for tk in missing:
                    try:
                        if len(missing) == 1:
                            close_ser = data["Close"] if "Close" in data.columns else None
                        else:
                            close_ser = (data[tk]["Close"]
                                         if tk in data.columns.get_level_values(0) else None)
                        if close_ser is not None:
                            valid = close_ser.dropna()
                            if not valid.empty:
                                prices[sym_map[tk]] = round(float(valid.iloc[-1]), 2)
                    except Exception:
                        continue
        except Exception:
            pass

    # ── METHOD 3: per-ticker history fallback (auto_adjust=False) ──────────────
    for tk, sym in sym_map.items():
        if sym in prices:
            continue
        base = tk[:-3]
        for sfx in [".NS", ".BO"]:
            try:
                t = _yf.Ticker(base + sfx)
                h = t.history(period="5d", interval="1d", auto_adjust=False)
                if h is not None and not h.empty and "Close" in h.columns:
                    valid = h["Close"].dropna()
                    if not valid.empty:
                        prices[sym] = round(float(valid.iloc[-1]), 2)
                        break
            except Exception:
                continue
    return prices

# ── Auto-refresh ───────────────────────────────────────────────────────────────
REFRESH_SEC = 300
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=REFRESH_SEC * 1000, key="dashboard_autorefresh")
except ImportError:
    pass

st.set_page_config(
    page_title="Swing Dashboard", page_icon="📈",
    layout="wide", initial_sidebar_state="expanded"
)

# ── Auth helpers ───────────────────────────────────────────────────────────────
def make_hash(password):
    """PBKDF2-HMAC-SHA256 with a random per-user salt (200k iterations).
    Stored format: pbkdf2$<salt_hex>$<hash_hex>"""
    import secrets as _secrets
    salt = _secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(),
                             bytes.fromhex(salt), 200_000)
    return f"pbkdf2${salt}${dk.hex()}"

def verify_hash(password, hashed_pw):
    """Verifies new pbkdf2 hashes AND legacy sha256 hashes, so existing
    accounts keep working. New registrations always get pbkdf2."""
    try:
        if str(hashed_pw).startswith("pbkdf2$"):
            _, salt, expected = str(hashed_pw).split("$", 2)
            dk = hashlib.pbkdf2_hmac("sha256", password.encode(),
                                     bytes.fromhex(salt), 200_000)
            import hmac as _hmac
            return _hmac.compare_digest(dk.hex(), expected)
        # Legacy: sha256 + static salt (accounts created before this upgrade)
        legacy = hashlib.sha256(str.encode(password + "swing_salt_99")).hexdigest()
        return legacy == hashed_pw
    except Exception:
        return False

def _cookie_secret():
    try:
        s = st.secrets.get("cookie_secret", "")
        if s:
            return str(s)
    except Exception:
        pass
    return "swing_cookie_fallback_v1"

def sign_uid(uid):
    """Cookie value = '<uid>.<hmac>' so nobody can forge another user's id."""
    import hmac as _hmac
    sig = _hmac.new(_cookie_secret().encode(), str(uid).encode(),
                    hashlib.sha256).hexdigest()[:24]
    return f"{uid}.{sig}"

def parse_signed_uid(token):
    """Returns the uid only if the signature is valid, else None."""
    import hmac as _hmac
    try:
        uid_str, sig = str(token).split(".", 1)
        expected = _hmac.new(_cookie_secret().encode(), uid_str.encode(),
                             hashlib.sha256).hexdigest()[:24]
        if _hmac.compare_digest(sig, expected):
            return int(uid_str)
    except Exception:
        pass
    return None

# ── Database ───────────────────────────────────────────────────────────────────
# Uses Neon DB (serverless Postgres). Neon is reachable from Streamlit Cloud and
# has no schema-resolution quirks. Connection params are passed as keyword
# are passed as explicit keyword args to psycopg2 — no URL string parsing.
# ==============================================================================

DB = "trades_v2.db"   # SQLite fallback (used if psycopg2 unavailable)

# ── NEON DB CONNECTION ─────────────────────────────────────────────────────────
def _load_pg_params():
    """Read Neon credentials from Streamlit Secrets (never hardcode in a
    public repo). If secrets are missing, returns None -> SQLite fallback."""
    try:
        return dict(
            host            = st.secrets["pg_host"],
            port            = int(st.secrets.get("pg_port", 5432)),
            dbname          = st.secrets.get("pg_dbname", "neondb"),
            user            = st.secrets["pg_user"],
            password        = st.secrets["pg_password"],
            sslmode         = "require",
            connect_timeout = 15,
        )
    except Exception:
        return None

_PG_PARAMS = _load_pg_params()
_USE_PG = _PG_PARAMS is not None

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    _USE_PG = False


def _pg_conn():
    """Open a Neon Postgres connection using explicit keyword params.
    Neon is serverless — the compute may be asleep and take a few seconds to
    wake on the first connection, so we retry several times with backoff."""
    last_err = None
    for _attempt in range(4):
        try:
            conn = psycopg2.connect(**_PG_PARAMS)
            conn.autocommit = False
            return conn
        except Exception as e:
            last_err = e
            time.sleep(1.0 * (_attempt + 1))   # 1s, 2s, 3s backoff for cold start
    raise last_err

def _q(sql):
    """Translate SQLite SQL → Postgres '%s' placeholders and 'INSERT OR REPLACE'.
    Schema qualification is handled directly in db() below."""
    if not _USE_PG:
        return sql
    s = sql.replace("?", "%s")
    s = s.replace("INSERT OR REPLACE INTO", "INSERT INTO")
    return s


_PG_SCHEMA_PREFIX = "public."
_PG_TABLES = ("users", "trades", "portfolio_history", "tg_config", "watchlist", "price_alerts", "trade_journal")


def _pg_qualify(sql):
    """No-op now. We rely on 'SET search_path TO public' (which Neon fully
    supports) rather than hardcoding public. prefixes. Kept as a function so
    callers don't need to change. Returns sql unchanged."""
    return sql


def db(sql, params=(), fetch=False):
    if _USE_PG:
        conn = _pg_conn()
        cur  = conn.cursor()
        # Neon fully supports search_path (unlike Supabase pooler). Set it so
        # bare table names resolve to public.* — this is the standard approach.
        try:
            cur.execute("SET search_path TO public")
        except Exception:
            pass
        # Also qualify explicitly as belt-and-suspenders.
        pg_sql = _pg_qualify(_q(sql))
        try:
            cur.execute(pg_sql, params)
            conn.commit()
        except Exception as e:
            conn.rollback()
            cur.close(); conn.close()
            # Surface the real SQL + error so the cause is visible, not redacted.
            raise RuntimeError(f"DB error on [{pg_sql}]: {type(e).__name__}: {e}") from e
        result = cur.fetchall() if fetch else None
        cur.close(); conn.close()
        return result
    else:
        conn = sqlite3.connect(DB)
        cur = conn.execute(sql, params)
        conn.commit()
        result = cur.fetchall() if fetch else None
        conn.close()
        return result

def init_db():
    if _USE_PG:
        conn = _pg_conn(); cur = conn.cursor()
        try:
            cur.execute("SET search_path TO public")
            conn.commit()
        except Exception:
            conn.rollback()
        # Create each table in its own transaction so one failure doesn't
        # abort the others (a failed statement poisons the whole transaction
        # in Postgres until rollback).
        table_ddls = [
            """CREATE TABLE IF NOT EXISTS users(
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL)""",
            """CREATE TABLE IF NOT EXISTS trades(
                id SERIAL PRIMARY KEY, user_id INTEGER, stock TEXT NOT NULL,
                quantity REAL NOT NULL, buy_at REAL NOT NULL, sell_at REAL,
                status TEXT DEFAULT 'Open',
                added_date TEXT DEFAULT to_char(CURRENT_DATE,'YYYY-MM-DD'),
                closed_date TEXT)""",
            """CREATE TABLE IF NOT EXISTS portfolio_history(
                id SERIAL PRIMARY KEY, user_id INTEGER, snapshot_date TEXT,
                total_invested REAL, current_value REAL)""",
            """CREATE TABLE IF NOT EXISTS tg_config(
                user_id INTEGER PRIMARY KEY, bot_token TEXT, chat_id TEXT)""",
            """CREATE TABLE IF NOT EXISTS watchlist(
                id SERIAL PRIMARY KEY, user_id INTEGER, stock TEXT NOT NULL,
                target_price REAL, notes TEXT,
                added_date TEXT DEFAULT to_char(CURRENT_DATE,'YYYY-MM-DD'))""",
            """CREATE TABLE IF NOT EXISTS price_alerts(
                id SERIAL PRIMARY KEY, user_id INTEGER, stock TEXT NOT NULL,
                condition TEXT NOT NULL, target_price REAL NOT NULL,
                status TEXT DEFAULT 'Active', note TEXT,
                created_date TEXT DEFAULT to_char(CURRENT_DATE,'YYYY-MM-DD'),
                triggered_date TEXT)""",
            """CREATE TABLE IF NOT EXISTS trade_journal(
                id SERIAL PRIMARY KEY, user_id INTEGER, stock TEXT NOT NULL,
                trade_date TEXT, direction TEXT, entry_price REAL, exit_price REAL,
                setup TEXT, rationale TEXT, emotion TEXT, outcome TEXT,
                lesson TEXT, rating INTEGER,
                created_date TEXT DEFAULT to_char(CURRENT_DATE,'YYYY-MM-DD'))""",
            """CREATE TABLE IF NOT EXISTS user_settings(
                user_id INTEGER PRIMARY KEY, theme TEXT)""",
        ]
        for ddl in table_ddls:
            try:
                cur.execute(ddl)
                conn.commit()
            except Exception:
                conn.rollback()
        cur.close(); conn.close()
    else:
        c = sqlite3.connect(DB)
        c.execute("""CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS trades(
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, stock TEXT NOT NULL,
            quantity REAL NOT NULL, buy_at REAL NOT NULL, sell_at REAL,
            status TEXT DEFAULT 'Open', added_date TEXT DEFAULT(date('now')),
            closed_date TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS portfolio_history(
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, snapshot_date TEXT,
            total_invested REAL, current_value REAL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS tg_config(
            user_id INTEGER PRIMARY KEY, bot_token TEXT, chat_id TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS watchlist(
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, stock TEXT NOT NULL,
            target_price REAL, notes TEXT, added_date TEXT DEFAULT(date('now')))""")
        c.execute("""CREATE TABLE IF NOT EXISTS price_alerts(
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, stock TEXT NOT NULL,
            condition TEXT NOT NULL, target_price REAL NOT NULL,
            status TEXT DEFAULT 'Active', note TEXT,
            created_date TEXT DEFAULT(date('now')), triggered_date TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS trade_journal(
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, stock TEXT NOT NULL,
            trade_date TEXT, direction TEXT, entry_price REAL, exit_price REAL,
            setup TEXT, rationale TEXT, emotion TEXT, outcome TEXT,
            lesson TEXT, rating INTEGER, created_date TEXT DEFAULT(date('now')))""")
        c.execute("""CREATE TABLE IF NOT EXISTS user_settings(
            user_id INTEGER PRIMARY KEY, theme TEXT)""")
        c.commit(); c.close()

def register_user(username, password):
    try:
        db("INSERT INTO users(username, password_hash) VALUES(?,?)",
           (username.lower(), make_hash(password)))
        return True
    except Exception:
        return False

def login_user(username, password):
    user = db("SELECT id, password_hash FROM users WHERE username=?",
              (username.lower(),), fetch=True)
    if user and verify_hash(password, user[0][1]):
        return user[0][0]
    return None


def get_user_theme(user_id):
    """Return the user's saved theme name, or None if not set / on error."""
    try:
        row = db("SELECT theme FROM user_settings WHERE user_id=?",
                 (user_id,), fetch=True)
        if row and row[0][0]:
            return row[0][0]
    except Exception:
        pass
    return None


def set_user_theme(user_id, theme):
    """Persist the user's chosen theme (upsert). Silent on failure."""
    try:
        if _USE_PG:
            db("INSERT INTO user_settings(user_id, theme) VALUES(?,?) "
               "ON CONFLICT (user_id) DO UPDATE SET theme=EXCLUDED.theme",
               (user_id, theme))
        else:
            db("INSERT OR REPLACE INTO user_settings(user_id, theme) "
               "VALUES(?,?)", (user_id, theme))
    except Exception:
        pass

def get_trades(user_id):
    if _USE_PG:
        conn = _pg_conn()
        try:
            cur = conn.cursor(); cur.execute("SET search_path TO public"); cur.close()
        except Exception:
            pass
        df = pd.read_sql_query(
            "SELECT * FROM trades WHERE user_id=%s ORDER BY id DESC",
            conn, params=(user_id,))
        conn.close()
        return df
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT * FROM trades WHERE user_id=? ORDER BY id DESC",
        conn, params=(user_id,))
    conn.close()
    return df

def get_history(user_id):
    if _USE_PG:
        conn = _pg_conn()
        try:
            cur = conn.cursor(); cur.execute("SET search_path TO public"); cur.close()
        except Exception:
            pass
        df = pd.read_sql_query(
            "SELECT * FROM portfolio_history WHERE user_id=%s ORDER BY snapshot_date",
            conn, params=(user_id,))
        conn.close()
        return df
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT * FROM portfolio_history WHERE user_id=? ORDER BY snapshot_date",
        conn, params=(user_id,))
    conn.close()
    return df

def get_watchlist(user_id):
    if _USE_PG:
        conn = _pg_conn()
        try:
            cur = conn.cursor(); cur.execute("SET search_path TO public"); cur.close()
        except Exception:
            pass
        df = pd.read_sql_query(
            "SELECT * FROM watchlist WHERE user_id=%s ORDER BY id DESC",
            conn, params=(user_id,))
        conn.close()
        return df
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT * FROM watchlist WHERE user_id=? ORDER BY id DESC",
        conn, params=(user_id,))
    conn.close()
    return df

def get_tg_config(user_id):
    rows = db("SELECT bot_token,chat_id FROM tg_config WHERE user_id=?",
              (user_id,), fetch=True)
    return rows[0] if rows else ("", "")

def save_tg_config(user_id, token, chat):
    if _USE_PG:
        db("INSERT INTO tg_config(user_id,bot_token,chat_id) VALUES(?,?,?) "
           "ON CONFLICT(user_id) DO UPDATE SET bot_token=EXCLUDED.bot_token, "
           "chat_id=EXCLUDED.chat_id",
           (user_id, token, chat))
    else:
        db("INSERT OR REPLACE INTO tg_config(user_id,bot_token,chat_id) VALUES(?,?,?)",
           (user_id, token, chat))

def add_trade(user_id, stock, qty, buy, sell=None):
    status = "Closed" if sell else "Open"
    closed = datetime.now().strftime("%Y-%m-%d") if sell else None
    db("INSERT INTO trades(user_id,stock,quantity,buy_at,sell_at,status,closed_date) VALUES(?,?,?,?,?,?,?)",
       (user_id, stock.upper().strip(), qty, buy, sell, status, closed))

def update_trade(tid, user_id, stock, qty, buy, sell, status):
    closed = datetime.now().strftime("%Y-%m-%d") if status == "Closed" else None
    db("UPDATE trades SET stock=?,quantity=?,buy_at=?,sell_at=?,status=?,closed_date=? WHERE id=? AND user_id=?",
       (stock.upper().strip(), qty, buy, sell, status, closed, tid, user_id))

def delete_trade(tid, user_id):
    db("DELETE FROM trades WHERE id=? AND user_id=?", (tid, user_id))

def close_trade(tid, user_id, sell):
    db("UPDATE trades SET sell_at=?,status='Closed',closed_date=? WHERE id=? AND user_id=?",
       (sell, datetime.now().strftime("%Y-%m-%d"), tid, user_id))

def save_snapshot(user_id, invested, value):
    """Save a daily portfolio snapshot. Non-critical — if it fails (e.g. DB
    hiccup), it must NOT crash the dashboard, so errors are swallowed."""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        db("DELETE FROM portfolio_history WHERE snapshot_date=? AND user_id=?", (today, user_id))
        db("INSERT INTO portfolio_history(user_id,snapshot_date,total_invested,current_value) VALUES(?,?,?,?)",
           (user_id, today, invested, value))
    except Exception:
        pass   # snapshot is non-essential; never block the dashboard on it

def add_watchlist(user_id, stock, target=None, notes=""):
    db("INSERT INTO watchlist(user_id,stock,target_price,notes) VALUES(?,?,?,?)",
       (user_id, stock.upper().strip(), target, notes))

def delete_watchlist_item(wid, user_id):
    db("DELETE FROM watchlist WHERE id=? AND user_id=?", (wid, user_id))

# ── Price Alert helpers ────────────────────────────────────────────────────────
def add_price_alert(user_id, stock, condition, target_price, note=""):
    """condition is 'above' or 'below'."""
    db("INSERT INTO price_alerts(user_id,stock,condition,target_price,note) "
       "VALUES(?,?,?,?,?)",
       (user_id, stock.upper().strip(), condition, float(target_price), note))

def get_price_alerts(user_id, status=None):
    if status:
        rows = db("SELECT id,stock,condition,target_price,status,note,"
                  "created_date,triggered_date FROM price_alerts "
                  "WHERE user_id=? AND status=? ORDER BY id DESC",
                  (user_id, status), fetch=True)
    else:
        rows = db("SELECT id,stock,condition,target_price,status,note,"
                  "created_date,triggered_date FROM price_alerts "
                  "WHERE user_id=? ORDER BY id DESC", (user_id,), fetch=True)
    return rows or []

def delete_price_alert(aid, user_id):
    db("DELETE FROM price_alerts WHERE id=? AND user_id=?", (aid, user_id))

def trigger_price_alert(aid, user_id):
    today = datetime.now().strftime("%Y-%m-%d")
    db("UPDATE price_alerts SET status='Triggered',triggered_date=? "
       "WHERE id=? AND user_id=?", (today, aid, user_id))

# ── Trade Journal helpers ──────────────────────────────────────────────────────
def add_journal_entry(user_id, stock, trade_date, direction, entry, exit_p,
                      setup, rationale, emotion, outcome, lesson, rating):
    db("INSERT INTO trade_journal(user_id,stock,trade_date,direction,entry_price,"
       "exit_price,setup,rationale,emotion,outcome,lesson,rating) "
       "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
       (user_id, stock.upper().strip(), trade_date, direction, entry, exit_p,
        setup, rationale, emotion, outcome, lesson, rating))

def get_journal_entries(user_id):
    rows = db("SELECT id,stock,trade_date,direction,entry_price,exit_price,setup,"
              "rationale,emotion,outcome,lesson,rating,created_date "
              "FROM trade_journal WHERE user_id=? ORDER BY id DESC",
              (user_id,), fetch=True)
    return rows or []

def delete_journal_entry(jid, user_id):
    db("DELETE FROM trade_journal WHERE id=? AND user_id=?", (jid, user_id))

# ── Session & Cookie Init ──────────────────────────────────────────────────────
from streamlit_cookies_controller import CookieController
controller = CookieController(key='app_cookies')

# Initialise DB and record connection status for the sidebar badge
_DB_STATUS = "sqlite"
_DB_ERROR = None
try:
    init_db()
    _DB_STATUS = "postgres" if _USE_PG else "sqlite"
except Exception as _db_e:
    _DB_ERROR = str(_db_e)
    # If Postgres was configured but failed, fall back to SQLite so the app
    # still loads (data won't persist, but the user isn't locked out).
    if _USE_PG:
        _USE_PG = False
        _DB_STATUS = "sqlite_fallback"
        try:
            init_db()
        except Exception:
            pass

# Ensure session state variables exist
for k, v in [("user_id", None), ("username", None), ("edit_id", None), ("close_id", None), ("del_id", None),
             ("last_refresh", None), ("last_auto_scan", 0.0), ("last_slow_scan", 0.0),
             ("_trade_hash", -1), ("sort_col", "stock"), ("sort_asc", False),
             ("signals_cache", None), ("sector_cache", None), ("picks_cache", None),
             ("outlook_cache", None), ("scanner_cache", None), ("trap_scan_cache", None),
             ("corp_actions_cache", None), ("selected_scanner_sector", "All Sectors"),
             ("custom_stocks_input", ""), ("active_page", "portfolio"),
             ("smc_scan_cache", None), ("vcp_scan_cache", None), ("rs_scan_cache", None),
             ("theme_scan_cache", None),
             ("etf_scan_cache", None), ("mf_search_results", []),
             ("mf_selected", None), ("mf_compare_list", []),
             ("_earnings_cache", None), ("_ipo_watch", []),
             ("first_render_done", False), ("_kickoff_scan", False),
             ("_scan_stage", "done"), ("_deep_stage", "sector"),
             ("_deep_running", False), ("_manual_deep_request", False),
             ("_manual_fast_request", False),
             ("_run_deep_now", False), ("_deep_progress", "done"),
             ("fast_interval_sec", 300), ("deep_interval_sec", 900),
             ("auto_fast", True), ("auto_deep", True),
             ("filter_status", "All"),
             ("filter_pnl", "All"), ("search", ""), ("theme", "Obsidian & Gold (Institutional)")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Auth Gate ──────────────────────────────────────────────────────────────────
if st.session_state.user_id is None:
    try:
        cookies = controller.getAll()
    except Exception:
        cookies = None

    # Bounded retry: the cookie component can return None on the first paint(s)
    # while it initialises. Retry a few times, then FALL THROUGH to the login
    # form with cookies treated as empty — never loop forever (that would keep
    # the page stuck on "Loading..." and the login screen would never appear).
    if cookies is None:
        _ck_tries = st.session_state.get("_ck_tries", 0)
        if _ck_tries < 3:
            st.session_state["_ck_tries"] = _ck_tries + 1
            st.info("Loading secure tunnel...")
            time.sleep(0.4)
            st.rerun()
        else:
            cookies = {}   # cookie store unavailable — proceed to manual login

    if cookies and cookies.get("swing_user_id"):
        try:
            cookie_uid = parse_signed_uid(cookies.get("swing_user_id"))
            if cookie_uid is None:
                raise ValueError("invalid cookie signature")
            st.session_state.user_id = cookie_uid
            user_row = db("SELECT username FROM users WHERE id=?",
                          (cookie_uid,), fetch=True)
            if user_row:
                st.session_state.username = user_row[0][0]
                st.session_state.first_render_done = False  # defer scans
                st.rerun()
        except Exception:
            pass

    st.markdown(
        "<h1 style='text-align:center;margin-top:5rem'>🔐 Quantitative Swing Dashboard</h1>",
        unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;color:gray'>Secure Multi-Tenant Gateway</p>",
        unsafe_allow_html=True)

    _, auth_col, _ = st.columns([1, 1.5, 1])
    with auth_col:
        tab_login, tab_signup = st.tabs(["Login", "Create Account"])

        with tab_login:
            with st.form("login_form"):
                l_user = st.text_input("Username")
                l_pass = st.text_input("Password", type="password")
                if st.form_submit_button("Access Terminal", width="stretch"):
                    uid = login_user(l_user, l_pass)
                    if uid:
                        st.session_state.user_id = uid
                        st.session_state.username = l_user
                        st.session_state.first_render_done = False  # defer scans
                        controller.set("swing_user_id", sign_uid(uid), max_age=604800)
                        st.success("Authenticated. Booting Engine...")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Invalid Username or Password")

        with tab_signup:
            with st.form("signup_form"):
                s_user = st.text_input("New Username")
                s_pass = st.text_input("New Password", type="password")
                if st.form_submit_button("Register Account", width="stretch"):
                    if len(s_user) < 3 or len(s_pass) < 4:
                        st.error("Username > 3 chars and Password > 4 chars required.")
                    else:
                        if register_user(s_user, s_pass):
                            st.success(f"✅ Account {s_user} registered!")
                        else:
                            st.error("❌ Username already exists.")
    st.stop()

# ==============================================================================
# MAIN APPLICATION (Only runs if Authenticated)
# ==============================================================================

# User ID strictly injected into all DB calls below
UID = st.session_state.user_id

# ── HARDCODED SIDEBAR TOGGLE ───────────────────────────────────────────────────
# Streamlit's native expand control is unreliable across versions and themes.
# This injects an always-present floating button that finds and clicks the real
# (possibly hidden) sidebar toggle via JS. Works on desktop AND mobile.
components.html("""
<script>
(function() {
    const doc = window.parent.document;
    if (doc.getElementById('hard-sidebar-toggle')) return;  // inject once

    const btn = doc.createElement('button');
    btn.id = 'hard-sidebar-toggle';
    btn.innerHTML = '☰';
    btn.title = 'Toggle sidebar';
    btn.style.cssText = [
        'position:fixed', 'top:10px', 'left:10px', 'z-index:2147483647',
        'width:42px', 'height:42px', 'border-radius:10px', 'border:none',
        'background:#d4af37', 'color:#000', 'font-size:20px', 'font-weight:800',
        'cursor:pointer', 'box-shadow:0 3px 14px rgba(0,0,0,.55)',
        'display:flex', 'align-items:center', 'justify-content:center',
        'transition:transform .15s ease'
    ].join(';');
    btn.onmouseover = () => btn.style.transform = 'scale(1.08)';
    btn.onmouseout  = () => btn.style.transform = 'scale(1)';

    btn.onclick = function() {
        // Try every known selector for the sidebar toggle, in priority order.
        const selectors = [
            '[data-testid="stSidebarCollapsedControl"] button',
            '[data-testid="stSidebarCollapsedControl"]',
            '[data-testid="collapsedControl"] button',
            '[data-testid="collapsedControl"]',
            '[data-testid="stSidebarCollapseButton"] button',
            '[data-testid="stSidebarCollapseButton"]',
            '[aria-label="Open sidebar"]',
            '[aria-label="Close sidebar"]'
        ];
        for (const sel of selectors) {
            const el = doc.querySelector(sel);
            if (el) { el.click(); return; }
        }
        // Last resort: toggle the sidebar width directly
        const sb = doc.querySelector('[data-testid="stSidebar"]');
        if (sb) {
            const hidden = sb.getAttribute('aria-expanded') === 'false'
                        || sb.style.transform.includes('-');
            sb.style.transform = hidden ? 'translateX(0)' : 'translateX(-100%)';
            sb.style.visibility = 'visible';
        }
    };
    doc.body.appendChild(btn);
})();
</script>
""", height=0)

THEMES = {
    # ── 1. The flagship: obsidian black + champagne gold, private-bank feel ──
    "Obsidian & Gold (Institutional)": {
        "bg": "#050608", "card": "rgba(13, 14, 18, 0.85)", "input": "#15171c",
        "border": "rgba(212, 175, 55, 0.18)",
        "text": "#fdfdfd", "muted": "#8e8e93",
        "green": "#10b981", "red": "#ef4444", "yellow": "#d4af37",
        "blue": "#3b82f6", "accent": "#d4af37", "card2": "#121419",
        "gradient": "linear-gradient(145deg, rgba(212,175,55,0.04) 0%, rgba(13,14,18,0.95) 35%, #050608 100%)",
        "glow": "rgba(212, 175, 55, 0.25)",
        "bg_fx": "radial-gradient(ellipse 80% 50% at 50% -20%, rgba(212,175,55,0.06), transparent)",
    },
    # ── 2. Bloomberg-terminal energy: near-black + signal orange ─────────────
    "Terminal Amber (Bloomberg)": {
        "bg": "#0a0a0a", "card": "rgba(18, 16, 12, 0.9)", "input": "#1a1813",
        "border": "rgba(255, 153, 0, 0.16)",
        "text": "#f5f0e8", "muted": "#9a917f",
        "green": "#33d17a", "red": "#ff5547", "yellow": "#ff9900",
        "blue": "#4da6ff", "accent": "#ff9900", "card2": "#161410",
        "gradient": "linear-gradient(160deg, rgba(255,153,0,0.05) 0%, rgba(18,16,12,0.95) 40%, #0a0a0a 100%)",
        "glow": "rgba(255, 153, 0, 0.22)",
        "bg_fx": "radial-gradient(ellipse 70% 45% at 80% -10%, rgba(255,153,0,0.05), transparent)",
    },
    # ── 3. Deep sapphire glassmorphism — frosted panels over midnight blue ───
    "Deep Sapphire (Glass)": {
        "bg": "#020617", "card": "rgba(15, 23, 42, 0.55)", "input": "#1e293b",
        "border": "rgba(56, 189, 248, 0.14)",
        "text": "#f8fafc", "muted": "#94a3b8",
        "green": "#10b981", "red": "#f43f5e", "yellow": "#f59e0b",
        "blue": "#0ea5e9", "accent": "#38bdf8", "card2": "rgba(30, 41, 59, 0.45)",
        "gradient": "linear-gradient(135deg, rgba(56,189,248,0.06) 0%, rgba(15,23,42,0.85) 45%, rgba(2,6,23,0.95) 100%)",
        "glow": "rgba(56, 189, 248, 0.25)",
        "bg_fx": "radial-gradient(ellipse 60% 40% at 20% -10%, rgba(56,189,248,0.08), transparent), radial-gradient(ellipse 50% 35% at 90% 10%, rgba(99,102,241,0.05), transparent)",
    },
    # ── 4. Emerald quant desk — money green on graphite ──────────────────────
    "Emerald Quant (Hedge Fund)": {
        "bg": "#060a08", "card": "rgba(11, 18, 14, 0.88)", "input": "#13201a",
        "border": "rgba(16, 185, 129, 0.16)",
        "text": "#f0fdf6", "muted": "#7e9a8c",
        "green": "#10b981", "red": "#f43f5e", "yellow": "#eab308",
        "blue": "#22d3ee", "accent": "#34d399", "card2": "#0e1812",
        "gradient": "linear-gradient(150deg, rgba(16,185,129,0.05) 0%, rgba(11,18,14,0.94) 40%, #060a08 100%)",
        "glow": "rgba(52, 211, 153, 0.22)",
        "bg_fx": "radial-gradient(ellipse 75% 50% at 50% -15%, rgba(16,185,129,0.06), transparent)",
    },
    # ── 5. Royal violet — premium fintech (Zerodha-dark x Stripe) ────────────
    "Royal Violet (Fintech)": {
        "bg": "#08060f", "card": "rgba(18, 13, 30, 0.88)", "input": "#1c1430",
        "border": "rgba(167, 139, 250, 0.16)",
        "text": "#faf8ff", "muted": "#9b8fc0",
        "green": "#34d399", "red": "#fb7185", "yellow": "#fbbf24",
        "blue": "#818cf8", "accent": "#a78bfa", "card2": "#150f26",
        "gradient": "linear-gradient(140deg, rgba(167,139,250,0.06) 0%, rgba(18,13,30,0.94) 40%, #08060f 100%)",
        "glow": "rgba(167, 139, 250, 0.25)",
        "bg_fx": "radial-gradient(ellipse 65% 45% at 30% -10%, rgba(167,139,250,0.07), transparent), radial-gradient(ellipse 50% 35% at 85% 5%, rgba(244,114,182,0.04), transparent)",
    },
    # ── 6. Carbon matrix — monochrome quant, teal data accents ───────────────
    "Carbon Matrix (Quant)": {
        "bg": "#09090b", "card": "rgba(18, 18, 20, 0.92)", "input": "#18181b",
        "border": "rgba(255, 255, 255, 0.07)",
        "text": "#fafafa", "muted": "#a1a1aa",
        "green": "#22c55e", "red": "#ff3366", "yellow": "#f59e0b",
        "blue": "#06b6d4", "accent": "#14b8a6", "card2": "#141416",
        "gradient": "linear-gradient(180deg, rgba(20,184,166,0.04) 0%, rgba(18,18,20,0.96) 35%, #09090b 100%)",
        "glow": "rgba(20, 184, 166, 0.20)",
        "bg_fx": "radial-gradient(ellipse 70% 45% at 50% -15%, rgba(20,184,166,0.05), transparent)",
    },
    # ── 7. Porcelain & Gold — private-bank light, warm ivory + deep gold ──────
    "Porcelain & Gold (Private Bank)": {
        "bg": "#f6f5f1", "card": "rgba(255, 255, 255, 0.85)", "input": "#ffffff",
        "border": "rgba(20, 33, 61, 0.12)",
        "text": "#14213d", "muted": "#6b7280",
        "green": "#059669", "red": "#dc2626", "yellow": "#b45309",
        "blue": "#1d4ed8", "accent": "#b8860b", "card2": "#efeee9",
        "gradient": "linear-gradient(145deg, rgba(184,134,11,0.06) 0%, rgba(255,255,255,0.92) 40%, #f6f5f1 100%)",
        "glow": "rgba(184, 134, 11, 0.18)",
        "bg_fx": "radial-gradient(ellipse 80% 50% at 50% -20%, rgba(184,134,11,0.05), transparent)",
    },
    # ── 8. Pearl & Emerald — wealth-desk light, cool pearl + sage emerald ─────
    "Pearl & Emerald (Atelier)": {
        "bg": "#f4f6f4", "card": "rgba(255, 255, 255, 0.82)", "input": "#ffffff",
        "border": "rgba(15, 82, 66, 0.12)",
        "text": "#1f2a24", "muted": "#6b7c74",
        "green": "#0f9d58", "red": "#d93838", "yellow": "#c08a00",
        "blue": "#2563eb", "accent": "#0f6e56", "card2": "#eaf0ec",
        "gradient": "linear-gradient(150deg, rgba(15,110,86,0.06) 0%, rgba(255,255,255,0.92) 40%, #f4f6f4 100%)",
        "glow": "rgba(15, 110, 86, 0.16)",
        "bg_fx": "radial-gradient(ellipse 75% 50% at 50% -15%, rgba(15,110,86,0.05), transparent)",
    },
    # ── 9. Platinum & Slate — quant light, cool platinum + clean blue ─────────
    "Platinum & Slate (Quant Light)": {
        "bg": "#f2f4f7", "card": "rgba(255, 255, 255, 0.85)", "input": "#ffffff",
        "border": "rgba(30, 41, 59, 0.12)",
        "text": "#0f172a", "muted": "#64748b",
        "green": "#059669", "red": "#e11d48", "yellow": "#d97706",
        "blue": "#0284c7", "accent": "#2563eb", "card2": "#e9edf2",
        "gradient": "linear-gradient(135deg, rgba(37,99,235,0.06) 0%, rgba(255,255,255,0.92) 45%, #f2f4f7 100%)",
        "glow": "rgba(37, 99, 235, 0.16)",
        "bg_fx": "radial-gradient(ellipse 60% 40% at 20% -10%, rgba(37,99,235,0.06), transparent), radial-gradient(ellipse 50% 35% at 90% 10%, rgba(2,132,199,0.04), transparent)",
    },
    # ── 10. Midnight Burgundy — reserve dark, wine + crimson accent ───────────
    "Midnight Burgundy (Reserve)": {
        "bg": "#0c0608", "card": "rgba(22, 12, 16, 0.85)", "input": "#1c1014",
        "border": "rgba(190, 18, 60, 0.20)",
        "text": "#fbf7f8", "muted": "#9a8a8e",
        "green": "#10b981", "red": "#fb7185", "yellow": "#e8b04b",
        "blue": "#60a5fa", "accent": "#c2334d", "card2": "#1a0e12",
        "gradient": "linear-gradient(145deg, rgba(190,18,60,0.05) 0%, rgba(22,12,16,0.95) 35%, #0c0608 100%)",
        "glow": "rgba(190, 18, 60, 0.22)",
        "bg_fx": "radial-gradient(ellipse 80% 50% at 50% -20%, rgba(190,18,60,0.07), transparent)",
    },
    # ── 11. Graphite & Copper — foundry dark, graphite + warm copper ──────────
    "Graphite & Copper (Foundry)": {
        "bg": "#0c0d0f", "card": "rgba(20, 21, 24, 0.88)", "input": "#1a1c1f",
        "border": "rgba(205, 127, 80, 0.16)",
        "text": "#f5f3f0", "muted": "#9b9690",
        "green": "#34d399", "red": "#f87171", "yellow": "#e0a458",
        "blue": "#60a5fa", "accent": "#cd7f50", "card2": "#141518",
        "gradient": "linear-gradient(150deg, rgba(205,127,80,0.05) 0%, rgba(20,21,24,0.94) 40%, #0c0d0f 100%)",
        "glow": "rgba(205, 127, 80, 0.20)",
        "bg_fx": "radial-gradient(ellipse 75% 50% at 50% -15%, rgba(205,127,80,0.06), transparent)",
    },
}

# --- Fail-safe to prevent KeyErrors when a saved theme name no longer exists ---
if st.session_state.theme not in THEMES:
    st.session_state.theme = "Obsidian & Gold (Institutional)"

def theme_css(t):
    glow  = t.get("glow", "rgba(255,255,255,0.1)")
    bg_fx = t.get("bg_fx", "none")
    # Pick readable text colour for filled (accent) buttons based on
    # the accent's luminance — fixes light-on-gold primary buttons on
    # light themes (e.g. "Execute Entry").
    def _on_accent(hexc):
        try:
            h = str(hexc).lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return "#0b0b0b" if (0.299*r + 0.587*g + 0.114*b) > 150 else "#ffffff"
        except Exception:
            return "#ffffff"
    on_accent = _on_accent(t.get("accent", "#ffffff"))
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {{
  --bg:{t['bg']}; --card:{t['card']}; --input:{t['input']};
  --border:{t['border']}; --text:{t['text']}; --muted:{t['muted']};
  --green:{t['green']}; --red:{t['red']}; --yellow:{t['yellow']};
  --blue:{t['blue']}; --accent:{t['accent']}; --card2:{t['card2']};
  --gradient:{t['gradient']}; --glow:{glow}; --on-accent:{on_accent};
}}

/* ═══ Base canvas with ambient light bloom ═══ */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
    background: var(--bg) !important; color: var(--text) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    font-feature-settings: 'cv02','cv03','cv04','cv11','ss01';
    letter-spacing: -0.011em;
    text-rendering: optimizeLegibility;
}}
/* Tabular figures — numbers align in neat columns (premium finance look) */
.sig-meta, .sig-price, .kpi-value, [data-testid="stMetricValue"],
.dataframe, code, .mono, [data-testid="stMetric"] {{
    font-feature-settings: 'tnum' 1, 'cv02','cv03';
    font-variant-numeric: tabular-nums;
}}
[data-testid="stAppViewContainer"]::before {{
    content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background: {bg_fx};
}}
[data-testid="stHeader"] {{ background: transparent !important; }}
#MainMenu, footer, header {{ display: none !important; }}
.block-container {{ padding-top: 1.5rem; padding-bottom: 3rem; max-width: 96%; }}
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 10px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--accent); }}

/* Numbers always tabular — institutional data discipline */
.card .val, table.t td, .sector-tbl td, .sig-meta, .pick-prices {{
    font-variant-numeric: tabular-nums;
}}

/* ═══ Title with animated accent underline ═══ */
.dash-title {{
    font-size: 2rem; font-weight: 800; padding-bottom: 1rem; margin-bottom: 1.5rem;
    display: flex; align-items: center; justify-content: space-between;
    letter-spacing: -0.03em; border-bottom: 1px solid var(--border);
    position: relative;
}}
.dash-title::after {{
    content: ""; position: absolute; bottom: -1px; left: 0; height: 2px; width: 180px;
    background: linear-gradient(90deg, var(--accent), transparent);
    animation: pulse-line 3s ease-in-out infinite;
}}
@keyframes pulse-line {{ 0%,100% {{ opacity: .5; width: 180px; }} 50% {{ opacity: 1; width: 280px; }} }}
.dash-title-text {{
    background: linear-gradient(to right, var(--text), var(--muted));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
.dash-title span.hl {{ color: var(--accent); -webkit-text-fill-color: var(--accent); }}
/* The title uses background-clip:text + transparent fill for its gradient. That
   works for letters but BREAKS emoji: an emoji is a colour bitmap, not a glyph
   the gradient can show through, so it renders as a solid dark box. This class
   opts the emoji out of the clip so it paints normally. */
.dash-title .ttl-icon {{
    -webkit-text-fill-color: initial; -webkit-background-clip: initial;
    background: none; color: initial;
    font-family: "Apple Color Emoji","Segoe UI Emoji","Noto Color Emoji",sans-serif;
}}

/* ═══ KPI cards — glass + top shimmer line + lift on hover ═══ */
.cards {{ display: flex; gap: 1.2rem; flex-wrap: wrap; margin-bottom: 2.5rem; }}
.card {{
    background: var(--gradient); border: 1px solid var(--border); border-radius: 16px;
    padding: 1.5rem; flex: 1; min-width: 160px; position: relative; overflow: hidden;
    box-shadow: 0 10px 30px -10px rgba(0,0,0,0.55);
    transition: all .4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
}}
.card::before {{
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    opacity: 0; transition: opacity .4s ease;
}}
.card:hover {{
    transform: translateY(-6px);
    box-shadow: 0 24px 48px -12px rgba(0,0,0,0.7), 0 0 24px var(--glow);
    border-color: var(--accent);
}}
.card:hover::before {{ opacity: 1; }}
.card .lbl {{
    font-size: .72rem; text-transform: uppercase; letter-spacing: .15em;
    color: var(--muted); margin-bottom: .5rem; font-weight: 700;
}}
.card .val {{ font-size: 1.6rem; font-weight: 800; color: var(--text); letter-spacing: -0.03em; }}
.card .sub {{ font-size: .8rem; color: var(--muted); margin-top: .4rem; font-weight: 600; }}

/* ═══ Section headers with gradient rail ═══ */
.green {{ color: var(--green) !important; }} .red {{ color: var(--red) !important; }}
.yellow {{ color: var(--yellow) !important; }} .blue {{ color: var(--blue) !important; }}
.sec {{
    font-size: .95rem; font-weight: 800; text-transform: uppercase;
    letter-spacing: .12em; color: var(--text); margin: 2.5rem 0 1.2rem;
    padding-left: 1rem; position: relative;
}}
.sec::before {{
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
    border-radius: 4px;
    background: linear-gradient(180deg, var(--accent), transparent);
}}

/* ═══ Tables — glass panel + accent header rail + row glow ═══ */
.tbl-wrap {{
    overflow-x: auto; background: var(--card); border: 1px solid var(--border);
    border-radius: 16px; box-shadow: 0 10px 30px -10px rgba(0,0,0,0.55);
    backdrop-filter: blur(16px); margin-bottom: 1.5rem;
}}
table.t, .sector-tbl {{ width: 100%; border-collapse: collapse; font-size: .85rem; }}
table.t th, .sector-tbl th {{
    background: var(--card2); color: var(--muted); text-transform: uppercase;
    letter-spacing: .08em; font-weight: 700; padding: 1.1rem 1rem; text-align: right;
    border-bottom: 1px solid var(--border); position: sticky; top: 0;
}}
table.t th.l, table.t td.l {{ text-align: left; }}
table.t td, .sector-tbl td {{
    padding: .95rem 1rem; border-bottom: 1px solid var(--border);
    text-align: right; color: var(--text); font-weight: 600;
    transition: background .2s ease;
}}
table.t tr:last-child td, .sector-tbl tr:last-child td {{ border-bottom: none; }}
table.t tr:hover td, .sector-tbl tr:hover td {{
    background: linear-gradient(90deg, transparent, var(--glow), transparent);
}}
table.t tr.row-profit td {{ box-shadow: inset 3px 0 0 var(--green); }}
table.t tr.row-loss td   {{ box-shadow: inset 3px 0 0 var(--red); }}

/* ═══ Badges with glow ═══ */
.pos {{ color: var(--green); font-weight: 800; }} .neg {{ color: var(--red); font-weight: 800; }}
.zero-cell {{ color: var(--muted) !important; }}
.badge {{
    display: inline-block; padding: .3rem .8rem; border-radius: 8px;
    font-size: .68rem; font-weight: 800; text-transform: uppercase; letter-spacing: .08em;
}}
.b-open {{ background: color-mix(in srgb, var(--yellow) 10%, transparent); color: var(--yellow);
    border: 1px solid color-mix(in srgb, var(--yellow) 35%, transparent);
    box-shadow: 0 0 12px color-mix(in srgb, var(--yellow) 12%, transparent); }}
.b-cl {{ background: rgba(16,185,129,.1); color: var(--green);
    border: 1px solid rgba(16,185,129,.35); box-shadow: 0 0 12px rgba(16,185,129,.12); }}
.b-cll {{ background: rgba(239,68,68,.1); color: var(--red);
    border: 1px solid rgba(239,68,68,.35); box-shadow: 0 0 12px rgba(239,68,68,.12); }}

/* ═══ Signal / pick / outlook cards — glass + animated entrance ═══ */
.sig-grid, .pick-grid, .outlook-grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 1.5rem; margin-top: 1rem;
}}
.sig-card, .pick-card, .outlook-card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 16px;
    padding: 1.5rem; transition: all .3s ease; backdrop-filter: blur(16px);
    box-shadow: 0 10px 20px -5px rgba(0,0,0,0.35);
    animation: card-in .45s ease both;
}}
@keyframes card-in {{ from {{ opacity: 0; transform: translateY(12px); }} to {{ opacity: 1; transform: none; }} }}
.sig-card:hover, .pick-card:hover {{
    transform: translateY(-5px); border-color: var(--accent);
    box-shadow: 0 18px 36px -8px rgba(0,0,0,0.55), 0 0 20px var(--glow);
}}
.sig-card.sell  {{ border-top: 3px solid var(--red); }}
.sig-card.avg   {{ border-top: 3px solid var(--yellow); }}
.sig-card.hold  {{ border-top: 3px solid var(--green); }}
.sig-card.watch {{ border-top: 3px solid var(--muted); }}
.pick-card      {{ border-top: 3px solid var(--accent); }}

.sig-action {{ font-size: .9rem; font-weight: 800; margin-bottom: .8rem;
    text-transform: uppercase; letter-spacing: .1em; }}
.sig-meta, .pick-sector {{ font-size: .8rem; color: var(--muted); font-weight: 600; }}
.sig-reason, .pick-prices {{ font-size: .88rem; margin-top: 1rem; color: var(--text); line-height: 1.65; }}
.sig-price, .pick-reason {{
    font-size: .82rem; margin-top: 1.2rem; padding-top: 1rem;
    border-top: 1px solid var(--border); font-weight: 600; color: var(--muted);
}}
.str-bar {{ height: 4px; border-radius: 2px; margin-top: 1rem; background: var(--input);
    overflow: hidden; }}
.str-fill {{ height: 100%; border-radius: 2px; transition: width .8s cubic-bezier(.22,1,.36,1); }}
.rr-warn {{ font-size: .75rem; color: var(--yellow); font-weight: 700; }}
.news-item {{
    padding: .6rem .9rem; border-left: 3px solid var(--accent); margin-bottom: .5rem;
    font-size: .85rem; background: var(--card2); border-radius: 0 8px 8px 0;
    transition: all .2s ease;
}}
.news-item:hover {{ border-left-width: 6px; background: var(--input); }}

/* ═══ Sidebar, inputs, buttons ═══ */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] div,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
[data-testid="stSidebar"] .stMarkdown div,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown span {{
    color: var(--text);
}}
[data-testid="stSidebar"] {{
    background: var(--card) !important; border-right: 1px solid var(--border);
    padding-top: 1rem;
}}
/* CRITICAL FIX — keep the sidebar expand ("maximize") control ALWAYS visible.
   Streamlit changed this test-id across versions, so we target every known
   variant. The backdrop-filter was removed from the sidebar above because it
   created a stacking context that hid this button after collapsing. */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[kind="headerNoPadding"][data-testid="baseButton-headerNoPadding"] {{
    display: flex !important; visibility: visible !important; opacity: 1 !important;
    z-index: 1000000 !important;
}}
/* The floating expand control when sidebar is collapsed */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {{
    position: fixed !important; top: .55rem !important; left: .55rem !important;
    background: var(--accent) !important; border-radius: 8px !important;
    padding: .3rem !important; box-shadow: 0 2px 12px rgba(0,0,0,.5) !important;
}}
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="collapsedControl"] svg {{
    color: #000 !important; fill: #000 !important; width: 1.5rem; height: 1.5rem;
}}
[data-testid="stSidebarCollapseButton"] svg {{ color: var(--text) !important; }}
/* The Streamlit top header bar can overlap the control — keep it transparent
   and non-blocking so the expand button is always clickable. */
[data-testid="stHeader"] {{
    background: transparent !important; z-index: 1 !important;
}}
/* Ensure dataframes scroll internally (both axes) and never clip results */
[data-testid="stDataFrame"] {{ overflow: auto !important; }}
[data-testid="stDataFrame"] > div {{ overflow: auto !important; max-width: 100% !important; }}
.stDataFrame [data-testid="stDataFrameResizable"] {{ overflow: auto !important; }}
/* Mobile: bigger tap target, sidebar takes most of the screen when open */
@media (max-width: 768px) {{
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {{
        top: .5rem !important; left: .5rem !important;
        padding: .45rem !important; transform: scale(1.2);
    }}
    [data-testid="stSidebar"] {{ min-width: 82vw !important; }}
}}
div[data-baseweb="input"], div[data-baseweb="select"],
[data-testid="stNumberInputContainer"] {{
    background-color: var(--input) !important; border: 1px solid var(--border) !important;
    border-radius: 10px !important; transition: border-color .2s ease !important;
}}
div[data-baseweb="input"]:focus-within, div[data-baseweb="select"]:focus-within,
[data-testid="stNumberInputContainer"]:focus-within {{
    border-color: var(--accent) !important; box-shadow: 0 0 0 2px var(--glow) !important;
}}
div[data-baseweb="input"] input, [data-testid="stNumberInputContainer"] input {{
    color: var(--text) !important; -webkit-text-fill-color: var(--text) !important;
    background-color: transparent !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; font-weight: 600 !important;
}}
button[data-testid="stNumberInputStepDown"], button[data-testid="stNumberInputStepUp"] {{
    background-color: var(--card2) !important; color: var(--text) !important; border: none !important;
}}
div[role="listbox"] {{ background-color: var(--card2) !important;
    border: 1px solid var(--border) !important; border-radius: 10px !important; }}
ul[role="listbox"] li {{ color: var(--text) !important; font-weight: 500 !important; }}
ul[role="listbox"] li[aria-selected="true"] {{
    background-color: var(--accent) !important; color: #000 !important; font-weight: 800 !important; }}

.stButton>button, [data-testid="stButton"] button,
[data-testid="baseButton-secondary"] {{
    background: var(--card2) !important; border: 1px solid var(--border) !important;
    color: var(--text) !important; border-radius: 10px !important; font-weight: 700 !important;
    letter-spacing: .05em !important; padding: .6rem 1.2rem !important;
    transition: all .3s ease !important;
}}
.stButton>button:hover, [data-testid="stButton"] button:hover,
[data-testid="baseButton-secondary"]:hover {{
    border-color: var(--accent) !important; background: var(--accent) !important;
    color: var(--on-accent) !important; box-shadow: 0 0 24px var(--glow) !important;
    transform: translateY(-1px) scale(1.01);
}}

/* ═══ Tabs — underline glide ═══ */
.stTabs [data-baseweb="tab-list"] {{
    background: transparent; gap: 2.2rem; padding: 0 .5rem;
    border-bottom: 1px solid var(--border);
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent; color: var(--muted); font-weight: 700; padding: 1.2rem 0;
    border: none; border-bottom: 3px solid transparent; text-transform: uppercase;
    letter-spacing: .08em; font-size: .82rem; transition: all .3s ease;
}}
.stTabs [data-baseweb="tab"]:hover {{ color: var(--text); }}
.stTabs [aria-selected="true"] {{
    background: transparent !important; color: var(--text) !important;
    border-bottom-color: var(--accent) !important;
    text-shadow: 0 0 18px var(--glow);
}}

/* ═══ Expanders ═══ */
[data-testid="stExpander"] {{
    background-color: var(--card) !important; border: 1px solid var(--border) !important;
    border-radius: 14px !important; margin-bottom: .8rem !important;
    backdrop-filter: blur(12px);
}}
[data-testid="stExpander"] summary p {{ font-weight: 700 !important; color: var(--text) !important; }}

/* ═══ Regime banner — live pulse dot + glass ═══ */
.refresh-badge {{
    display: inline-flex; align-items: center; gap: .5rem;
    background: rgba(16, 185, 129, 0.1); color: var(--green);
    padding: .4rem 1rem; border-radius: 30px; font-size: .72rem; font-weight: 800;
    border: 1px solid rgba(16, 185, 129, 0.4); letter-spacing: .1em;
    text-transform: uppercase; box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
}}
.refresh-badge::before {{
    content: ""; width: 7px; height: 7px; border-radius: 50%;
    background: var(--green); animation: live-pulse 1.8s ease-in-out infinite;
}}
@keyframes live-pulse {{
    0%, 100% {{ opacity: 1; box-shadow: 0 0 0 0 rgba(16,185,129,.5); }}
    50% {{ opacity: .6; box-shadow: 0 0 0 5px rgba(16,185,129,0); }}
}}
.regime-banner {{
    border-radius: 16px; padding: 1.2rem 1.8rem; display: flex; align-items: center;
    gap: 1.2rem; margin-bottom: 2.5rem; flex-wrap: wrap;
    box-shadow: 0 15px 35px -10px rgba(0,0,0,0.6); border: 1px solid var(--border);
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
}}

/* ════════════════════════════════════════════════════════════════════
   ✦ PREMIUM POLISH LAYER — refined typography, buttons, inputs, sidebar
   ════════════════════════════════════════════════════════════════════ */

/* Display serif for major headings — adds an editorial, private-bank feel.
   (Fraunces is already loaded in the main @import at the top of this stylesheet.) */
.dash-title, .sec {{
    font-family: 'Fraunces', Georgia, serif !important;
    font-weight: 600 !important; letter-spacing: -0.02em !important;
}}

/* Section headers get a refined gold tick + tighter rhythm */
.sec {{
    font-size: 1.35rem !important; font-weight: 600 !important;
    margin: 1.8rem 0 1.1rem !important; padding-left: .9rem !important;
    position: relative; line-height: 1.2;
}}
.sec::before {{
    content: ""; position: absolute; left: 0; top: 50%; transform: translateY(-50%);
    width: 4px; height: 70%; border-radius: 3px;
    background: linear-gradient(180deg, var(--accent), transparent);
}}

/* ✦ Buttons — premium gradient, lift, gold glow on hover */
.stButton > button, .stDownloadButton > button,
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button,
[data-testid="baseButton-secondary"] {{
    background: var(--card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-weight: 600 !important; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    letter-spacing: -0.01em !important;
    transition: all .25s cubic-bezier(0.175,0.885,0.32,1.275) !important;
}}
.stButton > button:hover, .stDownloadButton > button:hover,
[data-testid="stButton"] button:hover, [data-testid="stDownloadButton"] button:hover,
[data-testid="baseButton-secondary"]:hover {{
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px -8px var(--glow) !important;
}}
.stButton > button:active, .stDownloadButton > button:active {{
    transform: translateY(0) !important;
}}
/* Primary buttons (form submit, current-page nav) get the gold fill,
   app-wide — not just in the sidebar. */
button[kind*="primary"], [data-testid="baseButton-primary"],
[data-testid="baseButton-primaryFormSubmit"] {{
    background: linear-gradient(145deg, var(--accent), var(--accent)) !important;
    color: var(--on-accent) !important; border: none !important;
    box-shadow: 0 4px 20px -6px var(--glow) !important;
}}
button[kind*="primary"] p, button[kind*="primary"] span,
[data-testid="baseButton-primary"] p, [data-testid="baseButton-primary"] span,
[data-testid="baseButton-primaryFormSubmit"] p,
[data-testid="baseButton-primaryFormSubmit"] span {{
    color: var(--on-accent) !important;
}}
button[kind*="primary"]:hover, [data-testid="baseButton-primary"]:hover {{
    color: var(--on-accent) !important; filter: brightness(1.08);
}}

/* ✦ Inputs / selects — frosted with gold focus ring */
.stTextInput input, .stNumberInput input, .stDateInput input,
[data-baseweb="select"] > div, [data-baseweb="input"] > div {{
    background: var(--card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    transition: border-color .2s, box-shadow .2s !important;
}}
.stTextInput input:focus, .stNumberInput input:focus, .stDateInput input:focus {{
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--glow) !important;
}}

/* ✦ Tabs — underline slides, gold active */
.stTabs [data-baseweb="tab-list"] {{ gap: .3rem; border-bottom: 1px solid var(--border); }}
.stTabs [data-baseweb="tab"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    font-weight: 600 !important; color: var(--muted) !important;
    border-radius: 8px 8px 0 0 !important; transition: color .2s !important;
}}
.stTabs [aria-selected="true"] {{ color: var(--accent) !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: var(--accent) !important; }}

/* ✦ Sidebar — deeper glass, refined nav radio */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, rgba(0,0,0,.25), transparent), var(--card) !important;
    border-right: 1px solid var(--border) !important;
    backdrop-filter: blur(24px) !important; -webkit-backdrop-filter: blur(24px) !important;
}}
[data-testid="stSidebar"] .stRadio label {{
    transition: all .2s !important; border-radius: 8px !important;
    padding: .15rem .4rem !important;
}}
[data-testid="stSidebar"] .stRadio label:hover {{
    background: var(--glow) !important;
}}

/* ✦ Metric cards (st.metric) — give them the glass treatment */
[data-testid="stMetric"] {{
    background: var(--gradient) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important; padding: 1rem 1.2rem !important;
    box-shadow: 0 8px 24px -12px rgba(0,0,0,.5) !important;
    transition: transform .3s, border-color .3s !important;
}}
[data-testid="stMetric"]:hover {{
    transform: translateY(-3px) !important; border-color: var(--accent) !important;
}}
[data-testid="stMetricValue"] {{
    font-variant-numeric: tabular-nums !important; letter-spacing: -.02em !important;
}}

/* ✦ Dataframes — softer, rounded, bordered */
[data-testid="stDataFrame"] {{
    border: 1px solid var(--border) !important; border-radius: 12px !important;
    overflow: hidden !important;
}}

/* ✦ Expanders — refined */
[data-testid="stExpander"] {{
    border: 1px solid var(--border) !important; border-radius: 12px !important;
    background: var(--card) !important; overflow: hidden;
}}

/* ✦ Smooth fade-in for the whole view on load */
.block-container > div {{ animation: viewfade .5s ease both; }}
@keyframes viewfade {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; transform: none; }} }}

/* Respect reduced motion */
@media (prefers-reduced-motion: reduce) {{
    *, ::before, ::after {{ animation: none !important; transition: none !important; }}
}}

/* ═══ Sidebar buttons & popover triggers — text readable on every theme ═══ */
/* Blanket rule: ALL sidebar button labels follow --text (fixes invisible
   popover section names on light themes, regardless of Streamlit version). */
[data-testid="stSidebar"] button:not([kind*="primary"]) p,
[data-testid="stSidebar"] button:not([kind*="primary"]) span,
[data-testid="stSidebar"] button:not([kind*="primary"]) [data-testid="stMarkdownContainer"] p {{
    color: var(--text) !important;
}}
/* Any button whose kind CONTAINS "primary" (covers "primary" AND
   "primaryFormSubmit" in one rule — e.g. "Execute Entry") must read
   against its gold accent fill, not the generic --text colour. */
[data-testid="stSidebar"] button[kind*="primary"] p,
[data-testid="stSidebar"] button[kind*="primary"] span,
[data-testid="stSidebar"] button[kind*="primary"] [data-testid="stMarkdownContainer"] p {{
    color: var(--on-accent) !important;
}}
[data-testid="stSidebar"] button[kind*="primary"] {{
    background: linear-gradient(145deg, var(--accent), var(--accent)) !important;
    border: none !important;
}}
/* Sidebar popover TRIGGER buttons (e.g. "Portfolio ⌄") don't get Streamlit's
   normal button background — force the same themed card look as other
   sidebar buttons, on every theme (light or dark). */
[data-testid="stSidebar"] [data-testid="stPopover"] > div > button,
[data-testid="stSidebar"] div[data-baseweb="popover"] > button,
[data-testid="stSidebar"] button[data-testid*="Popover"],
[data-testid="stSidebar"] [data-testid*="Popover"] > button {{
    background: var(--card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}}
[data-testid="stSidebar"] [data-testid="stPopover"] > div > button:hover,
[data-testid="stSidebar"] div[data-baseweb="popover"] > button:hover,
[data-testid="stSidebar"] button[data-testid*="Popover"]:hover {{
    border-color: var(--accent) !important;
}}
/* Re-assert: PRIMARY buttons (current page / form submit) keep dark-on-accent */
[data-testid="stSidebar"] button[kind="primary"] p,
[data-testid="stSidebar"] button[kind="primary"] span,
[data-testid="stSidebar"] button[data-testid*="primary"] p,
[data-testid="stSidebar"] button[data-testid*="primary"] span {{
    color: var(--bg) !important;
}}

/* ═══ Light-theme contrast pack — text/icons that Streamlit leaves white ═══ */
/* (a) Popover PANEL page buttons live in a body-level portal, so the sidebar
   fix can't reach them. Force their labels to --text, but keep the current
   page (primary) button readable on its accent fill. */
[data-testid="stPopoverBody"] button:not([kind="primary"]) p,
[data-testid="stPopoverBody"] button:not([kind="primary"]) span,
div[data-baseweb="popover"] button:not([kind="primary"]) p,
div[data-baseweb="popover"] button:not([kind="primary"]) span {{
    color: var(--text) !important;
}}
[data-testid="stPopoverBody"] button[kind="primary"] p,
[data-testid="stPopoverBody"] button[kind="primary"] span,
div[data-baseweb="popover"] button[kind="primary"] p,
div[data-baseweb="popover"] button[kind="primary"] span {{
    color: var(--on-accent) !important;
}}
/* (b) Password reveal 👁 icon + any input-adornment icons → follow --text */
[data-testid="stTextInput"] button svg,
[data-testid="stTextInput"] button,
[data-baseweb="input"] button svg,
button[aria-label*="assword"] svg,
button[title*="assword"] svg {{
    color: var(--muted) !important; fill: var(--muted) !important;
}}
[data-testid="stTextInput"] button:hover svg,
[data-baseweb="input"] button:hover svg {{
    color: var(--text) !important; fill: var(--text) !important;
}}
/* (c) Checkbox / toggle / radio labels (e.g. "Show EMA 9/21" under the chart,
   "Ascending", scanner checkboxes) — force to --text on every theme. */
[data-testid="stCheckbox"] label,
[data-testid="stCheckbox"] label p,
[data-testid="stCheckbox"] label span,
[data-testid="stToggle"] label,
[data-testid="stToggle"] label p,
[data-testid="stToggle"] label span,
[data-testid="stRadio"] label p,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
[data-testid="stCheckbox"] [data-testid="stMarkdownContainer"] p,
[data-testid="stToggle"] [data-testid="stMarkdownContainer"] p,
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {{
    color: var(--text) !important;
}}
/* (d) Selectbox / multiselect chosen-value text (main area + sidebar) */
[data-baseweb="select"] div[value],
[data-baseweb="select"] span,
[data-testid="stMultiSelect"] span {{
    color: var(--text) !important;
}}
</style>
"""

# ── Price Fetcher & Logic ───────────────────────────────────────────────────────
_CACHE = {}
_TTL = 300


def fetch_price(symbol):
    clean = str(symbol).upper().strip()
    for sfx in [".NS", ".BO", ".NSE", ".BSE"]:
        if clean.endswith(sfx):
            clean = clean[:-len(sfx)]
    if clean in _CACHE and time.time() - _CACHE[clean][1] < _TTL:
        return _CACHE[clean][0]

    def _fast(t):
        try:
            fi = t.fast_info
        except Exception:
            return None
        for key in ("last_price", "lastPrice", "regularMarketPrice"):
            for getter in (lambda: fi.get(key) if hasattr(fi, "get") else None,
                           lambda: getattr(fi, key, None),
                           lambda: fi[key]):
                try:
                    v = getter()
                    if v is not None and not pd.isna(v) and float(v) > 0:
                        return float(v)
                except Exception:
                    pass
        return None

    for sfx in [".NS", ".BO"]:
        try:
            t = yf.Ticker(clean + sfx)
            v = _fast(t)
            if v is not None:
                p = round(v, 2)
                _CACHE[clean] = (p, time.time())
                return p
            # auto_adjust=False → ACTUAL close, not dividend/split back-adjusted
            h = t.history(period="5d", interval="1d", auto_adjust=False)
            if h is not None and not h.empty and "Close" in h.columns:
                lv = h["Close"].dropna()
                if not lv.empty:
                    p = round(float(lv.iloc[-1]), 2)
                    _CACHE[clean] = (p, time.time())
                    return p
        except Exception:
            continue
    return None


@st.cache_data(ttl=120, show_spinner=False)
def fetch_chart_data(symbol, period, interval):
    """Fetch OHLCV for the chart at any interval (intraday or daily).
    Yahoo intraday limits: 1m→7d max, 5m/15m→60d max, 1h→730d.
    auto_adjust=False for accurate actual prices. Returns DataFrame or None."""
    import yfinance as _yf
    clean = str(symbol).upper().strip()
    for sfx in [".NS", ".BO", ".NSE", ".BSE"]:
        if clean.endswith(sfx):
            clean = clean[:-len(sfx)]
    for sfx in [".NS", ".BO"]:
        for _attempt in range(2):
            try:
                t = _yf.Ticker(clean + sfx)
                df = t.history(period=period, interval=interval, auto_adjust=False)
                if df is not None and not df.empty and "Close" in df.columns:
                    return df.dropna(subset=["Close"])
            except Exception:
                pass
            try:
                df = _yf.download(clean + sfx, period=period, interval=interval,
                                  auto_adjust=False, progress=False, threads=False)
                if df is not None and not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    if "Close" in df.columns:
                        return df.dropna(subset=["Close"])
            except Exception:
                pass
            time.sleep(0.3)
    return None


# Valid (period, interval) combinations for Yahoo intraday data
_CHART_TIMEFRAMES = {
    "5 min":  ("5d",  "5m"),
    "15 min": ("1mo", "15m"),
    "1 hour": ("3mo", "1h"),
    "Daily":  ("6mo", "1d"),
    "Weekly": ("2y",  "1wk"),
}


def enrich(df):
    if df.empty:
        return df
    # Use cached price fetcher — avoids re-fetching on every Streamlit rerun
    symbols = tuple(sorted(df["stock"].unique().tolist()))
    prices  = _cached_prices(symbols)
    df = df.copy()
    df["cmp"] = df["stock"].map(prices)
    df["nse_label"] = "NSE:" + df["stock"]
    df["invested"] = df["quantity"] * df["buy_at"]
    df["current_amt"] = __import__("numpy").where(
        df["status"] == "Open",
        df["quantity"] * df["cmp"].fillna(df["buy_at"]),
        df["quantity"] * df["sell_at"].fillna(df["buy_at"])
    )
    df["total_amt"] = __import__("numpy").where(
        df["sell_at"].notna(),
        df["quantity"] * df["sell_at"],
        df["current_amt"]
    )
    df["profit"] = df["total_amt"] - df["invested"]
    df["profit_pct"] = (df["profit"] / df["invested"] * 100).round(2)
    return df


def calc_analytics(df):
    if df.empty:
        return {}
    closed = df[df["status"] == "Closed"]
    open_t = df[df["status"] == "Open"]
    total_closed = len(closed)
    wins   = len(closed[closed["profit"] > 0]) if not closed.empty else 0
    losses = len(closed[closed["profit"] < 0]) if not closed.empty else 0
    win_rate  = (wins / total_closed * 100) if total_closed > 0 else 0
    avg_win   = closed[closed["profit"] > 0]["profit"].mean() if wins > 0 else 0
    avg_loss  = abs(closed[closed["profit"] < 0]["profit"].mean()) if losses > 0 else 0
    exp = ((wins / total_closed if total_closed > 0 else 0) * avg_win) - \
          ((losses / total_closed if total_closed > 0 else 0) * avg_loss)
    gp = closed[closed["profit"] > 0]["profit"].sum() if wins > 0 else 0
    gl = abs(closed[closed["profit"] < 0]["profit"].sum()) if losses > 0 else 1
    max_dd = abs(
        (closed["profit"].cumsum() - closed["profit"].cumsum().expanding().max()).min()
    ) if not closed.empty else 0
    avg_hold = round(
        (pd.to_datetime(closed["closed_date"]) -
         pd.to_datetime(closed["added_date"])).dt.days.mean(), 1
    ) if not closed.empty and "closed_date" in closed.columns else 0
    sharpe = (closed["profit_pct"].mean() / closed["profit_pct"].std()
              if not closed.empty and closed["profit_pct"].std() > 0 else 0)
    return {
        "total_trades": len(df), "closed_trades": total_closed,
        "open_trades": len(open_t), "wins": wins, "losses": losses,
        "win_rate": round(win_rate, 1), "avg_win": round(avg_win, 0),
        "avg_loss": round(avg_loss, 0), "expectancy": round(exp, 0),
        "profit_factor": round(gp / gl, 2), "max_drawdown": round(max_dd, 0),
        "avg_hold_days": avg_hold, "sharpe": round(sharpe, 2)
    }


# ── Formatting helpers ─────────────────────────────────────────────────────────
def fi(v):   return f"₹{v:,.0f}"    if not pd.isna(v) else "—"
def fi2(v):  return f"₹{v:,.2f}"   if not pd.isna(v) else "—"
def fp(v):   return f"{'+' if v >= 0 else ''}{v:.2f}%" if not pd.isna(v) else "—"

def cv_cell(v, fn):
    if pd.isna(v):
        return f"<td>{fn(v)}</td>"
    if v > 0:
        return f'<td class="profit-cell pos">{fn(v)}</td>'
    if v < 0:
        return f'<td class="profit-cell neg">{fn(v)}</td>'
    return f'<td class="zero-cell">{fn(v)}</td>'

def badge(status, profit=None):
    if status == "Open":
        return '<span class="badge b-open">Open</span>'
    if profit is not None and profit < 0:
        return '<span class="badge b-cll">Closed ✗</span>'
    return '<span class="badge b-cl">Closed ✓</span>'

def card(lbl, val, sub="", cls=""):
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return f'<div class="card"><div class="lbl">{lbl}</div><div class="val {cls}">{val}</div>{sub_html}</div>'


# ── Chart helpers ──────────────────────────────────────────────────────────────
def _chart_colors():
    """Colors for Plotly pulled from the ACTIVE theme, so charts stay readable
    on light and dark themes alike."""
    t = THEMES.get(st.session_state.get("theme")) or next(iter(THEMES.values()))
    bg = str(t.get("bg", "#000000")).lstrip("#")
    try:
        r, g, b = int(bg[0:2], 16), int(bg[2:4], 16), int(bg[4:6], 16)
        is_light = (0.299 * r + 0.587 * g + 0.114 * b) > 140
    except Exception:
        is_light = False
    return {
        "text":  t.get("text", "#f8fafc"),
        "muted": t.get("muted", "#cbd5e1"),
        "grid":  t.get("border", "rgba(148,163,184,.2)"),
        "green": t.get("green", "#10b981"),
        "red":   t.get("red", "#ef4444"),
        "blue":  t.get("blue", "#3b82f6"),
        "yellow": t.get("yellow", "#f59e0b"),
        "pie_text": t.get("text", "#0f172a") if is_light else "#ffffff",
    }


def base_layout(fig, title):
    c = _chart_colors()
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=c["text"], weight="bold"), x=.01),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=c["muted"], size=11), margin=dict(l=8, r=8, t=45, b=8))
    fig.update_xaxes(gridcolor=c["grid"], zerolinecolor=c["grid"],
                     tickfont=dict(color=c["muted"]))
    fig.update_yaxes(gridcolor=c["grid"], zerolinecolor=c["grid"],
                     tickfont=dict(color=c["muted"]))
    return fig


def chart_alloc(df):
    c = _chart_colors()
    g = df.groupby("stock")["invested"].sum().reset_index()
    return base_layout(go.Figure(go.Pie(
        labels=g["stock"], values=g["invested"], hole=0.4,
        marker=dict(colors=px.colors.qualitative.Dark24,
                    line=dict(color="rgba(0,0,0,0)", width=0)),
        textinfo="percent+label", textfont=dict(size=11, color=c["pie_text"])
    )), "Portfolio Allocation")


def chart_pnl(df):
    c = _chart_colors()
    d = df.sort_values("profit")
    fig = base_layout(go.Figure(go.Bar(
        x=d["profit"], y=d["stock"], orientation="h",
        marker=dict(color=[c["red"] if v < 0 else c["green"] for v in d["profit"]],
                    line=dict(width=0)),
        text=[fp(p) for p in d["profit_pct"]],
        textposition="outside", textfont=dict(color=c["text"], size=10)
    )), "P&L by Stock")
    fig.update_layout(showlegend=False, margin=dict(l=8, r=55, t=45, b=8))
    fig.update_xaxes(tickprefix="₹")
    return fig


def chart_donut(df):
    c = _chart_colors()
    cnt = df["status"].value_counts().reset_index()
    cnt.columns = ["Status", "Count"]
    fig = base_layout(go.Figure(go.Pie(
        labels=cnt["Status"], values=cnt["Count"], hole=.6,
        marker=dict(
            colors=[{"Open": c["yellow"], "Closed": c["green"]}.get(s, c["muted"])
                    for s in cnt["Status"]],
            line=dict(color="rgba(0,0,0,0)", width=0)),
        textinfo="percent+value", textfont=dict(size=12, color=c["pie_text"])
    )), "Open vs Closed")
    fig.add_annotation(
        text=f"<b>{len(df)}</b><br><span style='font-size:10px'>TRADES</span>",
        font=dict(size=18, color=c["text"]), showarrow=False, x=.5, y=.5)
    return fig


def chart_growth(hist, cur_val, cur_inv):
    c = _chart_colors()
    today = datetime.now().strftime("%Y-%m-%d")
    rows = hist[["snapshot_date", "total_invested", "current_value"]].to_dict("records")
    if not hist.empty and hist.iloc[-1]["snapshot_date"] != today:
        rows.append({"snapshot_date": today,
                     "total_invested": cur_inv, "current_value": cur_val})
    elif hist.empty:
        rows = [{"snapshot_date": today,
                 "total_invested": cur_inv, "current_value": cur_val}]
    d = pd.DataFrame(rows)
    fig = base_layout(go.Figure([
        go.Scatter(x=pd.to_datetime(d["snapshot_date"]), y=d["current_value"],
                   name="Value", line=dict(color=c["green"], width=3),
                   fill="tozeroy", fillcolor="rgba(16,185,129,0.1)"),
        go.Scatter(x=pd.to_datetime(d["snapshot_date"]), y=d["total_invested"],
                   name="Invested", line=dict(color=c["blue"], width=2, dash="dash"))
    ]), "Portfolio Growth")
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(tickprefix="₹")
    return fig


# ── Signal card renderer ──────────────────────────────────────────────────────
def _fmt_rr(rr):
    if rr is None:
        return "—"
    if rr > 10:
        return f'<span class="rr-warn">⚠️ {rr} (verify ATR)</span>'
    if rr > 5:
        return f'<span style="color:#f59e0b;font-weight:700">{rr}</span>'
    return str(rr)


def render_signals(signals, theme_t):
    if not signals:
        st.info("No signals available.")
        return

    html = '<div class="sig-grid">'
    for s in signals:
        action = s.get("action", "")
        c = ("sell"  if "SELL"    in action else
             "avg"   if "AVERAGE" in action else
             "hold"  if "HOLD"    in action else "watch")
        clr = (theme_t["red"]    if c == "sell"  else
               theme_t["yellow"] if c == "avg"   else
               theme_t["green"]  if c == "hold"  else theme_t["muted"])

        cmp_v   = s.get("cmp")
        rsi_v   = s.get("rsi")
        pct     = s.get("pct_from_buy")
        target  = s.get("target")
        sl      = s.get("stop_loss")
        rr      = s.get("risk_reward")
        trend_v = s.get("trend", "—")
        macd_v  = s.get("macd_signal", "—")
        reason  = s.get("reason", "")
        strength = s.get("strength", 30)

        cmp_str = f"₹{cmp_v}" if cmp_v is not None else "—"
        rsi_str = str(rsi_v)  if rsi_v is not None else "—"
        pct_str = f"{pct:+.1f}%" if pct is not None else "—%"
        tgt_str = f"₹{target}" if target is not None else "—"
        sl_str  = f"₹{sl}"    if sl  is not None else "—"
        rr_html = _fmt_rr(rr)

        if c == "sell":
            ph = (f"🎯 Exit: {tgt_str} | 🛑 Re-entry: {sl_str}<br>"
                  f"📉 {trend_v} | MACD: {macd_v}")
        elif c == "avg":
            avg_p   = s.get("avg_price")
            new_avg = s.get("new_avg")
            new_sl  = s.get("new_sl")
            ph = (f"💰 Avg: {'₹'+str(avg_p) if avg_p else '—'} | "
                  f"New Avg: {'₹'+str(new_avg) if new_avg else '—'}<br>"
                  f"🛑 SL: {'₹'+str(new_sl) if new_sl else '—'} | 🎯 Target: {tgt_str}")
        else:
            ph = (f"🎯 Target: {tgt_str} | 🛑 SL: {sl_str}<br>"
                  f"📊 R:R {rr_html} | {trend_v}")

        # ── Build badges as clean single-line strings (no multi-line f-string
        #    expressions, which can break Streamlit's HTML rendering) ──────────
        badge_new = ""
        if s.get("limited_history"):
            badge_new = (f'<span style="font-size:.62rem;background:rgba(245,158,11,.15);'
                         f'color:#f59e0b;padding:.1rem .4rem;border-radius:4px;'
                         f'font-weight:700;margin-left:.3rem">🆕 {s.get("bars","")}d history</span>')

        badge_vcp = ""
        if s.get("vcp"):
            _vq = s.get("vcp_quality") or ""
            _vr = "▸READY" if s.get("vcp_ready") else ""
            badge_vcp = (f'<span style="font-size:.62rem;background:rgba(16,185,129,.18);'
                         f'color:#10b981;padding:.1rem .4rem;border-radius:4px;'
                         f'font-weight:700;margin-left:.3rem">🎯 VCP {_vq}{_vr}</span>')

        badge_rs = ""
        _rsr = s.get("rs_ratio")
        if s.get("rs_outperforming") and isinstance(_rsr, (int, float)):
            badge_rs = (f'<span style="font-size:.62rem;background:rgba(59,130,246,.18);'
                        f'color:#3b82f6;padding:.1rem .4rem;border-radius:4px;'
                        f'font-weight:700;margin-left:.3rem">💪 RS {_rsr:.2f}</span>')

        badges = badge_new + badge_vcp + badge_rs

        html += f"""
<div class="sig-card {c}">
  <div class="sig-action" style="color:{clr}">{action}</div>
  <div style="font-size:.9rem;font-weight:800;margin-bottom:.3rem">
    {s.get('stock','')}
    <span style="font-size:.7rem;color:var(--muted);font-weight:400">{s.get('sector','')}</span>
    {badges}
  </div>
  <div class="sig-meta">CMP {cmp_str} · RSI {rsi_str} · {pct_str}</div>
  <div class="sig-reason">{reason}</div>
  <div class="sig-price">{ph}</div>
  <div class="str-bar">
    <div class="str-fill" style="width:{strength}%;background:{clr}"></div>
  </div>
</div>"""

    # Collapse leading whitespace on each line — prevents Streamlit's markdown
    # parser from ever treating indented HTML as a code block (which would show
    # the raw <span> tags as literal text).
    html_clean = "\n".join(line.lstrip() for line in (html + "</div>").split("\n"))
    st.markdown(html_clean, unsafe_allow_html=True)


# ── Sector table renderer ─────────────────────────────────────────────────────
def render_sector(sdf, t):
    if sdf is None or sdf.empty:
        return

    def medal(rank):
        return "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "📊"

    rows = ""
    for _, r in sdf.iterrows():
        rs = r.get("rs_vs_nifty_1m", 0) or 0
        rs_clr = "#10b981" if rs > 0 else "#ef4444"
        avg_rsi = r["avg_rsi"]
        rsi_str = f"{avg_rsi:.0f}" if avg_rsi and not pd.isna(avg_rsi) else "—"
        rows += (
            f"<tr>"
            f"<td style='font-weight:700'>{medal(r['rank'])} #{int(r['rank'])}</td>"
            f"<td><b style='font-size:.9rem'>{r['sector']}</b></td>"
            f"<td style='color:var(--muted);font-size:.75rem'>{r['stocks']}</td>"
            f"<td style='text-align:center;font-weight:600'>{rsi_str}</td>"
            f"<td style='text-align:right'>"
            f"  <span class='{'pos' if r['avg_pct'] > 0 else 'neg'}'>"
            f"  {r['avg_pct']:+.1f}%</span></td>"
            f"<td style='text-align:right;font-size:.8rem'>"
            f"  <span style='color:{rs_clr};font-weight:700'>{rs:+.1f}%</span></td>"
            f"<td style='font-size:.8rem;font-weight:600'>"
            f"  {r.get('rrg_quadrant', '—')}</td>"
            f"<td>"
            f"  <div style='background:{t['input']};height:6px;width:100%;border-radius:4px'>"
            f"    <div style='background:{t['accent']};"
            f"         width:{min(r['momentum_score'] * 100, 100):.0f}%;"
            f"         height:6px;border-radius:4px'></div></div>"
            f"  <span style='font-size:.75rem;font-weight:600'>"
            f"  {r['momentum_score']:.2f}</span></td>"
            f"</tr>"
        )

    st.markdown(
        f'<div class="tbl-wrap"><table class="sector-tbl">'
        f'<thead><tr>'
        f'<th>Rank</th><th>Sector</th><th>Top Movers</th>'
        f'<th style="text-align:center">RSI</th>'
        f'<th style="text-align:right">1M Chg</th>'
        f'<th style="text-align:right">vs Nifty</th>'
        f'<th>RRG</th>'
        f'<th>Momentum</th>'
        f'</tr></thead><tbody>{rows}</tbody></table></div>',
        unsafe_allow_html=True
    )


def render_outlook(odf, t):
    if odf is None or odf.empty:
        return
    cards_html = ""
    for _, r in odf.iterrows():
        outlook = r["outlook"]
        clr = t["green"] if any(x in outlook for x in ["Bullish", "Power", "Strong"]) \
              else t["red"]
        avg_rsi = r.get("avg_rsi")
        rsi_str = f"{avg_rsi:.0f}" if avg_rsi and not pd.isna(avg_rsi) else "—"
        cards_html += (
            f"<div class='outlook-card'>"
            f"<div style='font-weight:700;margin-bottom:.3rem'>{r['sector']}</div>"
            f"<div style='color:{clr};font-weight:800;font-size:.9rem'>{outlook}</div>"
            f"<div class='sig-meta' style='margin-top:.4rem'>"
            f"Conf: {r['confidence']}% · Mom: {r['momentum']:.2f}<br>"
            f"RSI: {rsi_str} · Chg: {r['avg_pct']:+.1f}%</div>"
            f"</div>"
        )
    st.markdown(f'<div class="outlook-grid">{cards_html}</div>', unsafe_allow_html=True)


def render_picks(picks, t):
    if not picks:
        return
    cards_html = ""
    for p in picks:
        brd = t["green"] if p["score"] >= 70 else t["yellow"] if p["score"] >= 55 else t["muted"]
        cards_html += (
            f"<div class='pick-card' style='border-top-color:{brd}'>"
            f"<div style='font-weight:800'>{p['stock']} "
            f"<span class='pick-sector'>{p['sector']}</span></div>"
            f"<div style='font-size:.8rem;color:var(--muted);font-weight:600;margin-top:3px'>"
            f"CMP ₹{p['cmp']} · RSI {p['rsi']} · {p['trend']}</div>"
            f"<div class='pick-prices'>"
            f"🎯 Entry: ₹{p['entry']}<br>"
            f"🚀 Target: ₹{p['target']}<br>"
            f"🛑 SL: ₹{p['stop_loss']}<br>"
            f"📊 R:R: {_fmt_rr(p['risk_reward'])} · Score: {p['score']}</div>"
            f"<div class='pick-reason'>{p['reason']}</div>"
            f"</div>"
        )
    st.markdown(f'<div class="pick-grid">{cards_html}</div>', unsafe_allow_html=True)


# ── News renderer ─────────────────────────────────────────────────────────────
def render_news(news_list):
    if not news_list:
        st.info("No recent news found for your current holdings.")
        return
    for item in news_list:
        st.markdown(
            f'<div class="news-item">{item}</div>',
            unsafe_allow_html=True
        )


# ── Score dashboard ────────────────────────────────────────────────────────────
def render_score_dashboard():
    scores = [
        ("RSI (Wilder's)",          9, "adjust=False + explicit 100/0 edge case — matches TradingView"),
        ("MACD",                    9, "Single-pass crossover, adjust=False, histogram + momentum flags"),
        ("Bollinger Bands",         8, "bb_pos clamped [0,1], bandwidth + squeeze + breakout flags"),
        ("ATR",                     9, "Wilder's EWM smoothing — stops now match Zerodha/TV"),
        ("Supertrend",              9, "Numpy array loop, Wilder ATR(10), mult 2.5 for NSE swing"),
        ("VWAP",                    8, "20-day rolling VWAP + price_vs_vwap % deviation"),
        ("EMA / Trend",             8, "Slope flags (rising/flattening), momentum-fading label, EMA200 back"),
        ("Fibonacci",               8, "Swing-peak based via scipy with degenerate-swing fallback"),
        ("Chart Patterns",          8, "Neckline + Cup&Handle + vol gates"),
        ("Candlesticks",            8, "3-candle patterns, range normalization"),
        ("Signal RR Engine",        9, "Unified _calc_risk_params with PICK mode — zero phantom RR"),
        ("Sector Rotation",         8, "RRG quadrant + RS vs Nifty"),
        ("News Engine",             8, "yfinance v1.4 + RSS fallback"),
        ("Liquidity Gate",          8, "Soft gate: liquidity_ok flag, ⚠️ shown on signal, gated for new picks"),
        ("Unified Risk Engine",     9, "Scanner, picks, and portfolio signals all use one engine"),
        ("Bull/Bear Trap Scanner",  9, "5-factor confluence: geometry · volume quality · RSI extreme · Supertrend · reversal candle. Proactive sweep of full Nifty 500."),
        ("Smart Money Concepts",    8, "FVG · Order Blocks · Liquidity Pools · Premium/Discount · Displacement. NSE circuit-filter aware, ATR-normalised thresholds."),
        ("VCP (Volatility Contraction)", 8, "Minervini base detection: 2-4 tightening contractions, volume dry-up, pivot proximity, A+/A/B/C quality grading. Pivot-ready flag + dedicated scanner."),
        ("Relative Strength vs Nifty",   8, "IBD-style RS ratio (multi-period weighted) + 1-99 percentile rating across universe. Leaders boost conviction; laggards penalised. Dedicated RS Leaders ranking."),
    ]
    avg = sum(s[1] for s in scores) / len(scores)

    st.markdown(f"""
    <div style="background:var(--card);border:1px solid var(--border);border-radius:12px;
         padding:1.2rem 1.5rem;margin-bottom:1.5rem">
      <div style="font-size:.75rem;color:var(--muted);text-transform:uppercase;
           letter-spacing:.08em;margin-bottom:.5rem">signals.py — Overall Score</div>
      <div style="font-size:2.5rem;font-weight:800;color:var(--accent)">{avg:.1f}<span
           style="font-size:1rem;color:var(--muted);font-weight:400"> / 10</span></div>
      <div style="font-size:.8rem;color:var(--muted);margin-top:.3rem">
        Core engine v12 (Wilder ATR/RSI, numpy Supertrend, 20-day VWAP, swing-peak
        Fibonacci, unified risk engine) plus momentum stack: Trap detection, Smart
        Money Concepts, VCP base detection, and Relative Strength leadership ranking.
      </div>
    </div>
    """, unsafe_allow_html=True)

    rows_html = ""
    for name, score, note in scores:
        fill = min(score * 10, 100)
        clr = ("#10b981" if score >= 8 else "#f59e0b" if score >= 6 else "#ef4444")
        rows_html += f"""
<div style="margin-bottom:.8rem">
  <div style="display:flex;justify-content:space-between;align-items:center;
       margin-bottom:.3rem">
    <span style="font-size:.85rem;font-weight:600;color:var(--text)">{name}</span>
    <span style="font-size:.85rem;font-weight:800;color:{clr}">{score}/10</span>
  </div>
  <div style="background:var(--input);height:5px;border-radius:3px;margin-bottom:.3rem">
    <div style="background:{clr};width:{fill}%;height:5px;border-radius:3px"></div>
  </div>
  <div style="font-size:.75rem;color:var(--muted)">{note}</div>
</div>"""

    st.markdown(
        f'<div style="background:var(--card);border:1px solid var(--border);'
        f'border-radius:12px;padding:1.2rem 1.5rem">{rows_html}</div>',
        unsafe_allow_html=True
    )


# ── Load & Enrich Data ─────────────────────────────────────────────────────────
raw = get_trades(UID)
df  = enrich(raw) if not raw.empty else raw.copy()

if (st.session_state.last_refresh is None or
        (datetime.now() - st.session_state.last_refresh).seconds >= _TTL):
    st.session_state.last_refresh = datetime.now()

# ── Tiered background scan ─────────────────────────────────────────────────────
# FAST TIER (every 5 min):  portfolio signals + news (the core dashboard)
# DEEP TIER (every 15 min): sector rotation + picks + universe scanner + SMC scan
#
# CRITICAL LOAD-ORDER FIX:
# On first login the dashboard must RENDER FIRST, then scan. Otherwise the page
# blocks on the full deep scan (can be minutes) before login even completes.
# We use a `first_render_done` flag: the very first run after login skips all
# scanning, renders the dashboard immediately, and schedules a rerun. From the
# 2nd run onward the scans fire normally in the background.
_now = time.time()

open_raw = raw[raw["status"] == "Open"] if not raw.empty else pd.DataFrame()
_trade_hash = (hash(tuple(sorted(open_raw["id"].tolist())))
               if not open_raw.empty else 0)
_trades_changed = (_trade_hash != st.session_state._trade_hash)

# User-configurable intervals (seconds) and auto-scan toggles
_fast_interval = st.session_state.get("fast_interval_sec", 300)   # default 5 min
_deep_interval = st.session_state.get("deep_interval_sec", 900)   # default 15 min
_auto_fast = st.session_state.get("auto_fast", True)
_auto_deep = st.session_state.get("auto_deep", True)

if not st.session_state.get("first_render_done", False):
    # PASS 1 — first paint after login: render immediately, defer ALL scanning.
    st.session_state.first_render_done = True
    _fast_due = False
    _deep_due = False
    st.session_state._kickoff_scan = True
    st.session_state._scan_stage = "fast"
elif st.session_state.get("_scan_stage") == "fast":
    # PASS 2 — fast scan only (signals + news). Then kick off the deep sequence.
    _fast_due = True
    _deep_due = True
    st.session_state._deep_running = True
    st.session_state._deep_stage = "sector"
    st.session_state._scan_stage = "done"
elif st.session_state.get("_deep_running", False):
    # Deep scan mid-sequence — keep advancing (handled in post-render block).
    _fast_due = False
    _deep_due = True
else:
    # Steady state — scans fire only on their configured schedules (if auto on).
    st.session_state._scan_stage = "done"
    _fast_due = (_auto_fast and
                 (st.session_state.last_auto_scan == 0.0 or
                  (_now - st.session_state.last_auto_scan) >= _fast_interval))
    # Deep auto-trigger: start a fresh sequence when interval elapses
    _deep_elapsed = (st.session_state.last_slow_scan == 0.0 or
                     (_now - st.session_state.last_slow_scan) >= _deep_interval)
    _deep_due = False
    if _auto_deep and _deep_elapsed:
        st.session_state._deep_running = True
        st.session_state._deep_stage = "sector"
        _deep_due = True
    # Manual deep-scan request starts the sequence too
    if st.session_state.get("_manual_deep_request", False):
        st.session_state._manual_deep_request = False
        st.session_state._deep_running = True
        st.session_state._deep_stage = "sector"
        _deep_due = True
    # Manual fast-scan request
    if st.session_state.get("_manual_fast_request", False):
        st.session_state._manual_fast_request = False
        _fast_due = True

if _fast_due:
    n_open = len(open_raw)
    _spinner_msg = (f"🔔 Refreshing {n_open} signal{'s' if n_open!=1 else ''}…"
                    if n_open > 0 else "🔔 Refreshing market data…")
    with st.spinner(_spinner_msg):
        try:
            if _trades_changed or st.session_state.signals_cache is None:
                st.session_state.signals_cache = (
                    generate_signals(open_raw) if not open_raw.empty else [])
                st.session_state.news_cache = (
                    fetch_portfolio_news(open_raw) if not open_raw.empty else [])
                st.session_state._trade_hash = _trade_hash
            st.session_state.last_auto_scan = _now
        except Exception as _e:
            st.session_state.last_auto_scan = _now
            st.toast(f"⚠️ Signal refresh error: {_e}", icon="⚠️")

if _deep_due:
    # Deep scan is due — but we DEFER its execution to the very END of the script
    # (after the whole page renders) so it never blocks or interrupts your view.
    # The actual stage execution happens in the post-render block at the bottom.
    st.session_state._run_deep_now = True
else:
    st.session_state._run_deep_now = False


# ── Portfolio metrics ──────────────────────────────────────────────────────────
# Pre-initialise ALL portfolio totals so every downstream reference (KPI cards,
# sparkline, snapshots) is guaranteed a bound name regardless of data state.
odf = cdf = pd.DataFrame()
t_inv = t_cur = t_real = t_unreal = t_pnl = t_pnl_pct = 0
best = worst = "—"
if not df.empty:
    odf = df[df["status"] == "Open"]
    cdf = df[df["status"] == "Closed"]
    # Invested & current (portfolio) value reflect ONLY open positions — money
    # tied up in stocks you still hold. Sold/closed positions are excluded
    # because that capital has been freed up (their result lives in realized P&L).
    t_inv    = odf["invested"].sum()    if not odf.empty else 0
    t_cur    = odf["current_amt"].sum() if not odf.empty else 0
    t_real   = cdf["profit"].sum()   if not cdf.empty else 0   # realized (closed)
    t_unreal = odf["profit"].sum()   if not odf.empty else 0   # unrealized (open)
    t_pnl    = t_real + t_unreal       # total P&L across realized + unrealized
    # Return % is measured against invested capital. Use total cost basis
    # (open invested + the original cost of closed trades) so realized gains
    # aren't divided by zero when everything is sold.
    _closed_cost = cdf["invested"].sum() if not cdf.empty else 0
    _pnl_base = t_inv + _closed_cost
    t_pnl_pct = t_pnl / _pnl_base * 100 if _pnl_base > 0 else 0
    best  = df.loc[df["profit_pct"].idxmax(), "stock"]
    worst = df.loc[df["profit_pct"].idxmin(), "stock"]
    _today_snap = datetime.now().strftime("%Y-%m-%d")
    if st.session_state.get("_snap_done") != _today_snap:
        save_snapshot(UID, t_inv, t_cur)
        st.session_state._snap_done = _today_snap
else:
    odf = cdf = pd.DataFrame()
    t_inv = t_cur = t_real = t_unreal = t_pnl = t_pnl_pct = 0
    best = worst = "—"

# Load the user's saved theme from DB once per session (persists across refresh
# / restart). Runs only after login and only once, so live theme switches during
# the session aren't clobbered.
if not st.session_state.get("_theme_loaded") and st.session_state.get("user_id"):
    _saved_theme = get_user_theme(st.session_state.user_id)
    if _saved_theme and _saved_theme in THEMES:
        st.session_state.theme = _saved_theme
    st.session_state._theme_loaded = True

theme_t = THEMES[st.session_state.theme]
st.markdown(theme_css(theme_t), unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div style="font-size:.85rem;font-weight:800;color:var(--accent);'
        f'margin-bottom:1rem">👤 {st.session_state.username.upper()}</div>',
        unsafe_allow_html=True)

    # ── DB persistence status badge ────────────────────────────────────────────
    if _DB_STATUS == "postgres":
        st.markdown(
            '<div style="font-size:.7rem;font-weight:700;color:#10b981;'
            'background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.3);'
            'border-radius:6px;padding:.3rem .6rem;margin-bottom:.8rem">'
            '🟢 Postgres connected — data persists</div>',
            unsafe_allow_html=True)
    elif _DB_STATUS == "sqlite_fallback":
        st.markdown(
            '<div style="font-size:.7rem;font-weight:700;color:#ef4444;'
            'background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);'
            'border-radius:6px;padding:.3rem .6rem;margin-bottom:.8rem">'
            '🔴 Postgres failed — using temporary storage. '
            'Check DATABASE_URL secret.</div>',
            unsafe_allow_html=True)
        if _DB_ERROR:
            with st.expander("⚠️ DB error detail"):
                st.code(_DB_ERROR[:300])
    else:
        st.markdown(
            '<div style="font-size:.7rem;font-weight:700;color:#f59e0b;'
            'background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.3);'
            'border-radius:6px;padding:.3rem .6rem;margin-bottom:.8rem">'
            '🟡 Local storage (data resets on restart). '
            'Add DATABASE_URL to persist.</div>',
            unsafe_allow_html=True)

    if st.button("🚪 Logout", width="stretch"):
        controller.set("swing_user_id", "", max_age=0)
        st.session_state.clear()
        st.rerun()

    st.markdown("<hr style='margin:1rem 0;border-color:var(--border)'>",
                unsafe_allow_html=True)
    st.markdown('<div style="font-size:.8rem;font-weight:800;letter-spacing:.05em">'
                '🎨 UI THEME</div>', unsafe_allow_html=True)

    new_theme = st.selectbox(
        "Theme", list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.theme),
        label_visibility="collapsed")
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        if st.session_state.get("user_id"):
            set_user_theme(st.session_state.user_id, new_theme)
        st.rerun()

    st.markdown("<hr style='margin:.8rem 0;border-color:var(--border)'>",
                unsafe_allow_html=True)
    st.markdown('<div style="font-size:.8rem;font-weight:800;letter-spacing:.05em;'
                'margin-bottom:.5rem">🗺 NAVIGATION</div>', unsafe_allow_html=True)

    NAV_GROUPS = {
        "📊 Portfolio": [
            ("📋 Overview",          "portfolio"),
            ("📊 Charts",            "analytics"),
            ("📈 Stock Chart",       "chart"),
            ("📐 Metrics",           "metrics"),
            ("📤 Export",            "export"),
        ],
        "🔔 Signals & Alerts": [
            ("🔔 Active Signals",    "signals"),
            ("🪤 Trap Scanner",      "traps"),
            ("🏦 Smart Money (SMC)", "smc"),
            ("📐 VCP Scanner",       "vcp"),
        ],
        "🔄 Market Intelligence": [
            ("🔄 Sector Rotation",   "sector"),
            ("🎯 Theme Scanner",     "themes"),
            ("💪 RS Leaders",        "rs"),
            ("🌌 Universe Scanner",  "scanner"),
            ("🚀 Scanner 2.0",       "scanner2"),
            ("📊 Market Breadth",    "breadth"),
            ("📰 Market News",       "news"),
            ("🔬 Custom Screener",   "screener"),
            ("📅 Corporate Actions", "corp_actions"),
            ("📆 Earnings Calendar", "earnings"),
            ("🆕 IPO Tracker",       "ipo"),
        ],
        "💰 Funds & ETFs": [
            ("📈 ETF Tracker",       "etfs"),
            ("🏛 Mutual Funds",      "mutual_funds"),
        ],
        "🛡 Risk & Sizing": [
            ("🧮 Position Sizing",   "sizing"),
            ("🛡 Risk Dashboard",    "risk"),
        ],
        "🛠 Tools": [
            ("👁 Watchlist",         "watchlist"),
            ("🔔 Price Alerts",      "alerts"),
            ("📓 Trade Journal",     "journal"),
            ("🎯 Signal Scores",     "scores"),
        ],
    }
    # Flat list for radio
    nav_labels = [label for group in NAV_GROUPS.values() for label, _ in group]
    nav_keys   = [key   for group in NAV_GROUPS.values() for _, key   in group]

    if "active_page" not in st.session_state:
        st.session_state.active_page = "portfolio"

    # Popover nav: each section is a popover button; its pages live inside the
    # floating panel. Click a section to open it, then click a page. The current
    # page renders as a highlighted (primary) button.
    cur_page = st.session_state.active_page
    for group_label, items in NAV_GROUPS.items():
        with st.popover(group_label, use_container_width=True):
            for label, key in items:
                if st.button(label, key=f"nav_{key}", width="stretch",
                             type=("primary" if key == cur_page else "secondary")):
                    st.session_state.active_page = key
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.1rem;font-weight:800;color:var(--accent);'
                'margin-bottom:.8rem">⚡ Trade Entry</div>', unsafe_allow_html=True)

    em   = st.session_state.edit_id is not None
    erow = (raw[raw["id"] == st.session_state.edit_id].iloc[0]
            if em and not raw.empty else None)
    if em:
        st.markdown(
            '<div style="background:rgba(99,102,241,.15);border:1px solid rgba(99,102,241,.4);'
            'border-radius:8px;padding:.5rem;font-size:.8rem;color:var(--accent);'
            'margin-bottom:1rem;font-weight:700">✏️ Editing trade</div>',
            unsafe_allow_html=True)

    with st.form("trade_form", clear_on_submit=True):
        s_in   = st.text_input("Stock Symbol",
                               value=erow["stock"] if erow is not None else "",
                               placeholder="CDSL, IRFC…")
        q_in   = st.number_input("Quantity", min_value=1, step=1,
                                 value=int(erow["quantity"]) if erow is not None else 1)
        b_in   = st.number_input("Buy At ₹", min_value=0.01, step=0.05,
                                 value=float(erow["buy_at"]) if erow is not None else 0.01,
                                 format="%.2f")
        sel_in = st.number_input(
            "Sell At ₹ (optional)", min_value=0.0, step=0.05,
            value=float(erow["sell_at"]) if (erow is not None and erow["sell_at"]) else 0.0,
            format="%.2f")

        if st.form_submit_button(
                "💾 Update Trade" if em else "➕ Execute Entry", width="stretch",
                type="primary"):
            if not s_in.strip():
                st.error("Symbol required")
            elif b_in <= 0:
                st.error("Buy price must be > 0")
            else:
                sv = sel_in if sel_in > 0 else None
                if em:
                    update_trade(st.session_state.edit_id, UID, s_in, q_in, b_in,
                                 sv, "Closed" if sv else "Open")
                    st.session_state.edit_id = None
                    st.success("Updated!")
                else:
                    add_trade(UID, s_in, q_in, b_in, sv)
                    st.success(f"Added {s_in.upper()}")
                _CACHE.clear()
                st.session_state.last_auto_scan = 0.0
                st.rerun()

    if em and st.button("✖ Cancel Edit", width="stretch"):
        st.session_state.edit_id = None
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:.8rem;font-weight:800;letter-spacing:.05em">'
                '🔍 FILTERS</div>', unsafe_allow_html=True)
    st.session_state.filter_status = st.selectbox(
        "Status", ["All", "Open", "Closed"],
        index=["All", "Open", "Closed"].index(st.session_state.filter_status),
        label_visibility="collapsed")
    st.session_state.filter_pnl = st.selectbox(
        "P&L", ["All", "Profitable", "Loss"],
        index=["All", "Profitable", "Loss"].index(st.session_state.filter_pnl),
        label_visibility="collapsed")
    st.session_state.search = st.text_input(
        "Search", value=st.session_state.search,
        placeholder="Search symbol…", label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:.8rem;font-weight:800;letter-spacing:.05em">'
                '⚙️ SCAN CONTROLS</div>', unsafe_allow_html=True)

    _interval_opts = {"5 min": 300, "15 min": 900, "30 min": 1800}
    _interval_labels = list(_interval_opts.keys())

    # Core (fast) scan controls
    st.markdown('<div style="font-size:.72rem;color:var(--muted);font-weight:700;'
                'margin:.5rem 0 .2rem">⚡ Core scan (signals · news · prices)</div>',
                unsafe_allow_html=True)
    fc1, fc2 = st.columns([1, 1])
    with fc1:
        st.session_state.auto_fast = st.toggle(
            "Auto", value=st.session_state.auto_fast, key="toggle_fast")
    with fc2:
        _cur_fast = next((k for k, v in _interval_opts.items()
                          if v == st.session_state.fast_interval_sec), "5 min")
        _sel_fast = st.selectbox("Every", _interval_labels,
                                 index=_interval_labels.index(_cur_fast),
                                 key="sel_fast", label_visibility="collapsed")
        st.session_state.fast_interval_sec = _interval_opts[_sel_fast]
    if st.button("⚡ Scan Core Now", width="stretch"):
        st.session_state._manual_fast_request = True
        _cached_prices.clear()
        st.rerun()

    # Deep scan controls
    st.markdown('<div style="font-size:.72rem;color:var(--muted);font-weight:700;'
                'margin:.7rem 0 .2rem">🔄 Deep scan (sector · universe · SMC · trap · VCP · RS)</div>',
                unsafe_allow_html=True)
    dc1, dc2 = st.columns([1, 1])
    with dc1:
        st.session_state.auto_deep = st.toggle(
            "Auto", value=st.session_state.auto_deep, key="toggle_deep")
    with dc2:
        _cur_deep = next((k for k, v in _interval_opts.items()
                          if v == st.session_state.deep_interval_sec), "15 min")
        _sel_deep = st.selectbox("Every", _interval_labels,
                                 index=_interval_labels.index(_cur_deep),
                                 key="sel_deep", label_visibility="collapsed")
        st.session_state.deep_interval_sec = _interval_opts[_sel_deep]
    if st.button("🔄 Scan Deep Now", width="stretch"):
        st.session_state._manual_deep_request = True
        st.rerun()

    # Status / countdown
    _elapsed_fast = time.time() - st.session_state.last_auto_scan
    _elapsed_slow = time.time() - st.session_state.last_slow_scan
    _nxt_fast = max(0, int((st.session_state.fast_interval_sec - _elapsed_fast) // 60))
    _nxt_slow = max(0, int((st.session_state.deep_interval_sec - _elapsed_slow) // 60))
    _stage_names = {"sector": "Sector rotation", "universe": "Universe scan",
                    "smc": "SMC setups", "traps": "Trap scan",
                    "vcp": "VCP bases", "rs": "RS leaders"}
    if st.session_state.get("_deep_running", False):
        _cur = st.session_state.get("_deep_stage", "sector")
        _deep_status = f'⏳ {_stage_names.get(_cur, _cur)}…'
    else:
        _deep_status = f'{_nxt_slow}m' if st.session_state.auto_deep else 'manual'
    _fast_status = f'{_nxt_fast}m' if st.session_state.auto_fast else 'manual'
    st.markdown(
        f'<div style="font-size:.68rem;color:var(--muted);padding-top:.5rem;'
        f'font-weight:600;line-height:1.6">'
        f'⚡ core: {_fast_status} · 🔄 deep: {_deep_status}</div>',
        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:.8rem;font-weight:800;letter-spacing:.05em">'
                '📱 TELEGRAM</div>', unsafe_allow_html=True)

    # Load from DB first; fall back to st.secrets; cache in session_state
    # so values survive within-session navigation without a DB re-read.
    if "tg_tok_saved" not in st.session_state or "tg_cid_saved" not in st.session_state:
        db_tok, db_cid = get_tg_config(UID)
        if not db_tok:
            try:
                db_tok = st.secrets.get("telegram_bot_token", "")
                db_cid = st.secrets.get("telegram_chat_id", "")
            except Exception:
                db_tok = db_cid = ""
        st.session_state.tg_tok_saved = db_tok or ""
        st.session_state.tg_cid_saved = db_cid or ""

    saved_tok = st.session_state.tg_tok_saved
    saved_cid = st.session_state.tg_cid_saved

    tg_tok = st.text_input("Bot Token", value=saved_tok, type="password")
    tg_cid = st.text_input("Chat ID",   value=saved_cid)
    if st.button("💾 Save Config", width="stretch"):
        save_tg_config(UID, tg_tok, tg_cid)
        st.session_state.tg_tok_saved = tg_tok
        st.session_state.tg_cid_saved = tg_cid
        st.success("Saved!")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="dash-title">'
    '<div class="dash-title-text"><span class="ttl-icon">📈</span> Quantitative <span class="hl">Swing Dashboard</span></div>'
    '<span class="refresh-badge">⚡ SIGNALS LIVE · 🔄 SECTOR LIVE</span>'
    '</div>',
    unsafe_allow_html=True)

market = _get_market_regime_safe()
regime = market.get("regime", "Unknown")

rc_map = {
    "Strong Bull":  ("rgba(16,185,129,.15)",  "#10b981", "border:1px solid rgba(16,185,129,.4)"),
    "Bull":         ("rgba(16,185,129,.1)",   "#10b981", "border:1px solid rgba(16,185,129,.2)"),
    "Bull Pullback":("rgba(245,158,11,.15)",  "#f59e0b", "border:1px solid rgba(245,158,11,.4)"),
    "Strong Bear":  ("rgba(239,68,68,.15)",   "#ef4444", "border:1px solid rgba(239,68,68,.4)"),
    "Bear":         ("rgba(239,68,68,.1)",    "#ef4444", "border:1px solid rgba(239,68,68,.2)"),
    "Bear Rally":   ("rgba(245,158,11,.15)",  "#f59e0b", "border:1px solid rgba(245,158,11,.4)"),
}
rc_bg, rc_clr, rc_border = rc_map.get(
    regime, ("rgba(148,163,184,.1)", "#94a3b8", "border:1px solid rgba(148,163,184,.3)"))

indices_html = ""
_idx_items = market.get("indices", {})
for name, d in _idx_items.items():
    price = d.get("price")
    chg   = d.get("chg_pct", 0)
    price_str = f"₹{price:,.0f}" if price else "—"
    if name == "India VIX":
        chg_clr = "var(--red)" if chg > 0 else "var(--green)"
    else:
        chg_clr = "var(--green)" if chg > 0 else "var(--red)"
    indices_html += (
        f'<span style="color:var(--text);font-size:.8rem;padding:0 .8rem;'
        f'border-right:1px solid rgba(255,255,255,.1)">'
        f'{name} <b>{price_str}</b> '
        f'<span style="color:{chg_clr};font-weight:700">{chg:+.2f}%</span></span>'
    )
if not _idx_items:
    # Indices failed to load — show a clear note instead of a blank banner
    indices_html = (
        '<span style="color:var(--muted);font-size:.78rem;padding:0 .8rem">'
        '📡 Index data loading… (Yahoo Finance may be rate-limited; '
        'refreshes automatically)</span>'
    )

sup_str = f"₹{market.get('support'):,.0f}" if market.get("support") else "—"
res_str = f"₹{market.get('resistance'):,.0f}" if market.get("resistance") else "—"

st.markdown(
    f'<div class="regime-banner" style="background:{rc_bg};{rc_border};'
    f'backdrop-filter:blur(10px)">'
    f'<span style="color:{rc_clr};font-weight:800;font-size:.9rem;'
    f'white-space:nowrap;letter-spacing:.05em">'
    f'🌐 {regime.upper()} (CONF: {market.get("confidence","—")}%)</span>'
    f'{indices_html}'
    f'<span style="color:var(--muted);font-size:.75rem;white-space:nowrap;'
    f'padding-left:.5rem;font-weight:600">'
    f'SUP: {sup_str} | RES: {res_str} | '
    f'RSI {market.get("nifty_rsi","—")} | RISK: {market.get("risk_level","—")}'
    f'</span></div>',
    unsafe_allow_html=True)

# ── KPI cards ──────────────────────────────────────────────────────────────────
# ── Portfolio value sparkline (30-day) for the KPI card ───────────────────────
_spark_html = ""
try:
    _now_ts = time.time()
    if (_now_ts - st.session_state.get("_hist_ts", 0)) > 600:
        st.session_state["_hist_df"] = get_history(UID)
        st.session_state["_hist_ts"] = _now_ts
    _hh = st.session_state.get("_hist_df")
    if _hh is not None and len(_hh) >= 3:
        _vals = _hh["current_value"].astype(float).tail(30).tolist()
        if t_cur:
            _vals.append(float(t_cur))
        _vmin, _vmax = min(_vals), max(_vals)
        _rng = (_vmax - _vmin) or 1.0
        _n = len(_vals)
        _pts = " ".join(
            f"{(i/(_n-1))*100:.1f},{26 - ((v - _vmin)/_rng)*24:.1f}"
            for i, v in enumerate(_vals))
        _sclr = "#10b981" if _vals[-1] >= _vals[0] else "#ef4444"
        _spark_html = (f'<svg width="100%" height="28" viewBox="0 0 100 28" '
                       f'preserveAspectRatio="none" style="margin-top:.3rem">'
                       f'<polyline points="{_pts}" fill="none" stroke="{_sclr}" '
                       f'stroke-width="2" stroke-linejoin="round"/></svg>')
except Exception:
    _spark_html = ""

pnl_c = "green" if t_pnl >= 0 else "red"
r_c   = "green" if t_real >= 0 else "red"
u_c   = "green" if t_unreal >= 0 else "red"

st.markdown(
    '<div class="cards">'
    + card("Total Invested",  fi(t_inv),    "",          "blue")
    + card("Portfolio Value", fi(t_cur), _spark_html, "blue")
    + card("Total P&L",       fi(t_pnl),    fp(t_pnl_pct), pnl_c)
    + card("Realized P&L",    fi(t_real),   "",          r_c)
    + card("Unrealized P&L",  fi(t_unreal), "",          u_c)
    + card("Open Trades",     str(len(odf)), "Active",   "yellow")
    + card("Closed Trades",   str(len(cdf)), "Historical","green" if len(cdf) > 0 else "")
    + card("Best Trade 🏆",   best,          "",         "green")
    + card("Worst Trade 📉",  worst,         "",         "red")
    + '</div>',
    unsafe_allow_html=True)

# ── Tabs ────────────────────────────────────────────────────────────────────────
# ── Page routing — driven by sidebar navigation ────────────────────────────────
_page = st.session_state.get("active_page", "portfolio")

# ── Portfolio ────────────────────────────────────────────────────────────────
if _page == 'portfolio':
    if df.empty:
        st.info("No trades yet. Use the sidebar to execute an entry.")
    else:
        fdf = df.copy()
        if st.session_state.filter_status != "All":
            fdf = fdf[fdf["status"] == st.session_state.filter_status]
        if st.session_state.filter_pnl == "Profitable":
            fdf = fdf[fdf["profit"] > 0]
        elif st.session_state.filter_pnl == "Loss":
            fdf = fdf[fdf["profit"] < 0]
        if st.session_state.search.strip():
            fdf = fdf[fdf["stock"].str.upper().str.contains(
                st.session_state.search.upper())]

        sort_opts = {
            "Stock": "stock", "Qty": "quantity", "Buy At": "buy_at",
            "CMP": "cmp", "Invested": "invested",
            "P&L ₹": "profit", "P&L %": "profit_pct"
        }
        sc1, sc2 = st.columns([3, 1])
        with sc1:
            sort_key = st.selectbox(
                "Sort by", list(sort_opts.keys()),
                index=list(sort_opts.values()).index(st.session_state.sort_col)
                if st.session_state.sort_col in sort_opts.values() else 0,
                label_visibility="collapsed")
            sort_col = sort_opts[sort_key]
        with sc2:
            asc = st.toggle("⬆ Ascending", value=st.session_state.sort_asc)
            st.session_state.sort_asc = asc

        st.session_state.sort_col = sort_col
        if sort_col in fdf.columns:
            fdf = fdf.sort_values(sort_col, ascending=asc, na_position="last")

        st.markdown(
            f'<div class="sec">Open Positions & History ({len(fdf)})</div>',
            unsafe_allow_html=True)

        rows_html = ""
        for _, r in fdf.iterrows():
            row_cls = ("row-profit" if r.get("profit", 0) > 0
                       else "row-loss" if r.get("profit", 0) < 0 else "row-neutral")
            cmp_cell = (
                '<td class="zero-cell">—</td>' if pd.isna(r.get("cmp"))
                else f'<td class="pos">{fi2(r["cmp"])}</td>' if r["cmp"] > r["buy_at"]
                else f'<td class="neg">{fi2(r["cmp"])}</td>' if r["cmp"] < r["buy_at"]
                else f'<td>{fi2(r["cmp"])}</td>'
            )
            cur_cell = (
                '<td>—</td>' if pd.isna(r.get("current_amt", 0))
                else f'<td class="pos">{fi(r["current_amt"])}</td>'
                     if r["current_amt"] > r["invested"]
                else f'<td class="neg">{fi(r["current_amt"])}</td>'
                     if r["current_amt"] < r["invested"]
                else f'<td>{fi(r["current_amt"])}</td>'
            )
            rows_html += (
                f"<tr class='{row_cls}'>"
                f"<td class='l'><span class='nse-lbl'>{r.get('nse_label','')}</span></td>"
                f"<td class='l'><b style='font-size:.9rem'>{r['stock']}</b><br>"
                f"<span style='font-size:.7rem;color:var(--muted)'>"
                f"{get_sector(r['stock'])} · {r.get('added_date','')}</span></td>"
                f"<td>{int(r['quantity'])}</td>"
                f"<td>{fi2(r['buy_at'])}</td>"
                f"{cmp_cell}"
                f"<td>{'—' if pd.isna(r.get('sell_at')) else fi2(r['sell_at'])}</td>"
                f"<td>{fi(r['invested'])}</td>"
                f"{cur_cell}"
                f"{cv_cell(r.get('profit', 0), fi)}"
                f"{cv_cell(r.get('profit_pct', 0), fp)}"
                f"<td>{badge(r['status'], r.get('profit', 0))}</td>"
                f"</tr>"
            )

        st.markdown(
            f'<div class="tbl-wrap"><table class="t"><thead><tr>'
            f'<th class="l">NSE</th><th class="l">Asset</th>'
            f'<th>Qty</th><th>Entry</th><th>CMP</th><th>Exit</th>'
            f'<th>Invested</th><th>Value</th><th>P&L ₹</th><th>P&L %</th>'
            f'<th>Status</th></tr></thead><tbody>{rows_html}</tbody></table></div>',
            unsafe_allow_html=True)

        st.markdown('<div class="sec">Manage Positions</div>', unsafe_allow_html=True)
        opts = [f"{r['id']} — {r['stock']}" for _, r in fdf.iterrows()]
        if opts:
            ca, cb, cc, cd = st.columns([3, 1, 1, 1])
            with ca:
                sel_id = int(
                    st.selectbox("Select Trade ID", opts,
                                 label_visibility="collapsed").split(" — ")[0])
            with cb:
                if st.button("✏️ Modify", width="stretch"):
                    st.session_state.edit_id = sel_id
                    st.rerun()
            with cc:
                if st.button("🔒 Close Pos", width="stretch"):
                    st.session_state.close_id = sel_id
                    st.rerun()
            with cd:
                if st.button("🗑 Drop", width="stretch"):
                    st.session_state.del_id = sel_id
                    st.rerun()

        if st.session_state.close_id:
            st.markdown("---")
            st.markdown("**Execute Close — Confirm Exit Price**")
            sp = st.number_input("Exit Price ₹", min_value=0.01, step=0.05, format="%.2f")
            x1, x2 = st.columns(2)
            with x1:
                if st.button("✅ Confirm Exit", width="stretch"):
                    close_trade(st.session_state.close_id, UID, sp)
                    st.session_state.close_id = None
                    st.rerun()
            with x2:
                if st.button("✖ Abort", width="stretch"):
                    st.session_state.close_id = None
                    st.rerun()

        if st.session_state.del_id:
            st.markdown("---")
            st.warning(
                f"Drop trade ID #{st.session_state.del_id}? This is irreversible.")
            y1, y2 = st.columns(2)
            with y1:
                if st.button("🗑 Confirm Drop", width="stretch"):
                    delete_trade(st.session_state.del_id, UID)
                    st.session_state.del_id = None
                    st.rerun()
            with y2:
                if st.button("✖ Abort", width="stretch"):
                    st.session_state.del_id = None
                    st.rerun()

# ── Charts / Analytics ───────────────────────────────────────────────────────
elif _page == 'analytics':
    if df.empty:
        st.info("Execute trades to populate visualization models.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(chart_alloc(df), use_container_width=True)
        with c2:
            st.plotly_chart(chart_donut(df), use_container_width=True)
        st.plotly_chart(chart_pnl(df), use_container_width=True)
        st.plotly_chart(
            chart_growth(get_history(UID), t_cur, t_inv),
            use_container_width=True)

        # ── Enhanced analytics (Batch 2) ──────────────────────────────────────
        st.markdown("<hr style='border-color:var(--border);margin:1.5rem 0'>",
                    unsafe_allow_html=True)
        st.markdown('<div class="sec">📊 Performance Breakdown</div>',
                    unsafe_allow_html=True)

        closed = df[df["status"] == "Closed"].copy()

        ec1, ec2 = st.columns(2)

        # 1. P&L by sector (realized, closed trades)
        with ec1:
            if not closed.empty:
                closed["sector"] = closed["stock"].apply(get_sector)
                sec_pnl = closed.groupby("sector")["profit"].sum().sort_values()
                colors = ["#ef4444" if v < 0 else "#10b981" for v in sec_pnl.values]
                bfig = go.Figure(go.Bar(
                    x=sec_pnl.values, y=sec_pnl.index, orientation="h",
                    marker_color=colors))
                bfig.update_layout(
                    title="Realized P&L by Sector", height=300,
                    margin=dict(l=10, r=10, t=40, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=theme_t.get("text", "#fff")))
                bfig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
                st.plotly_chart(bfig, use_container_width=True)
            else:
                st.info("No closed trades yet for sector P&L.")

        # 2. Win/Loss distribution
        with ec2:
            if not closed.empty:
                wins = len(closed[closed["profit"] > 0])
                losses = len(closed[closed["profit"] < 0])
                be = len(closed[closed["profit"] == 0])
                pfig = go.Figure(go.Pie(
                    labels=["Wins", "Losses", "Breakeven"],
                    values=[wins, losses, be], hole=0.5,
                    marker_colors=["#10b981", "#ef4444", "#f59e0b"]))
                pfig.update_layout(
                    title="Win / Loss Distribution", height=300,
                    margin=dict(l=10, r=10, t=40, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=theme_t.get("text", "#fff")))
                st.plotly_chart(pfig, use_container_width=True)
            else:
                st.info("No closed trades yet for win/loss split.")

        # 3. Best & worst trades
        if not closed.empty:
            st.markdown("##### 🏆 Best & Worst Closed Trades")
            closed_sorted = closed.sort_values("profit_pct", ascending=False)
            bw1, bw2 = st.columns(2)
            with bw1:
                st.markdown("**🟢 Top 5 Winners**")
                top5 = closed_sorted.head(5)[["stock", "profit", "profit_pct"]].copy()
                top5.columns = ["Stock", "P&L ₹", "P&L %"]
                st.dataframe(top5, width="stretch", hide_index=True)
            with bw2:
                st.markdown("**🔴 Top 5 Losers**")
                bot5 = closed_sorted.tail(5)[["stock", "profit", "profit_pct"]].copy()
                bot5 = bot5.sort_values("P&L %" if "P&L %" in bot5.columns else "profit_pct")
                bot5.columns = ["Stock", "P&L ₹", "P&L %"]
                st.dataframe(bot5, width="stretch", hide_index=True)

        # 4. Holding period distribution
        if not closed.empty and "closed_date" in closed.columns:
            try:
                closed["hold_days"] = (
                    pd.to_datetime(closed["closed_date"]) -
                    pd.to_datetime(closed["added_date"])).dt.days
                valid_hold = closed["hold_days"].dropna()
                if not valid_hold.empty:
                    st.markdown("##### ⏱ Holding Period Distribution")
                    hfig = go.Figure(go.Histogram(
                        x=valid_hold, marker_color="#3b82f6", nbinsx=20))
                    hfig.update_layout(
                        height=260, margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title="Days held", yaxis_title="Trades",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color=theme_t.get("text", "#fff")))
                    hfig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
                    hfig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
                    st.plotly_chart(hfig, use_container_width=True)
                    avg_hold = valid_hold.mean()
                    st.caption(f"Average holding period: {avg_hold:.0f} days")
            except Exception:
                pass

# ── Active Signals ───────────────────────────────────────────────────────────
elif _page == 'signals':
    st.markdown(
        '<div class="sec">Active Portfolio Signals & Risk Management</div>',
        unsafe_allow_html=True)

    s1, s2 = st.columns([2, 1])
    with s1:
        st.caption("🤖 Neural background scan refreshes every 15 minutes.")
    with s2:
        if st.button("📲 Push to Telegram", width="stretch",
                     disabled=not bool(saved_tok and saved_cid)):
            if st.session_state.signals_cache is not None:
                with st.spinner("🤖 Compiling Telegram report..."):
                    msg_payload = build_telegram_message(
                        st.session_state.signals_cache,
                        st.session_state.sector_cache
                        if st.session_state.sector_cache is not None else pd.DataFrame(),
                        st.session_state.picks_cache
                        if st.session_state.picks_cache is not None else []
                    )
                    news = st.session_state.news_cache or []
                    if news:
                        msg_payload += "\n\n🌍 <b>LATEST HOLDINGS NEWS</b>\n"
                        msg_payload += "\n".join(news[:8])
                    ok = send_telegram(saved_tok, saved_cid, msg_payload)
                    if ok:
                        st.success("✅ Broadcast successful!")
                    else:
                        st.error("❌ Broadcast failed. Check token/chat ID.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec">🌍 Live Portfolio News</div>',
                unsafe_allow_html=True)

    with st.expander("📰 Latest Headlines for Active Holdings", expanded=False):
        open_raw = raw[raw["status"] == "Open"] if not raw.empty else pd.DataFrame()
        if not open_raw.empty:
            col_news1, col_news2 = st.columns([3, 1])
            with col_news2:
                force_news = st.button("🔄 Refresh News", width="stretch")
            if force_news:
                with st.spinner("Fetching latest headlines..."):
                    st.session_state.news_cache = fetch_portfolio_news(open_raw)

            render_news(st.session_state.news_cache or [])
        else:
            st.info("No active trades. Add a trade to see related news.")

    if st.session_state.signals_cache is not None:
        nc = {"SELL": 0, "AVERAGE": 0, "HOLD": 0, "WATCH": 0}
        for s in st.session_state.signals_cache:
            for k in nc:
                if k in s.get("action", ""):
                    nc[k] += 1

        st.markdown(
            f'<div style="display:flex;gap:.8rem;margin:.5rem 0 1rem">'
            f'<span style="background:rgba(239,68,68,.15);color:#ef4444;'
            f'padding:.3rem .8rem;border-radius:6px;font-size:.8rem;font-weight:800;'
            f'border:1px solid rgba(239,68,68,.3)">🔴 SELL: {nc["SELL"]}</span>'
            f'<span style="background:rgba(245,158,11,.15);color:#f59e0b;'
            f'padding:.3rem .8rem;border-radius:6px;font-size:.8rem;font-weight:800;'
            f'border:1px solid rgba(245,158,11,.3)">🟡 AVERAGE: {nc["AVERAGE"]}</span>'
            f'<span style="background:rgba(16,185,129,.15);color:#10b981;'
            f'padding:.3rem .8rem;border-radius:6px;font-size:.8rem;font-weight:800;'
            f'border:1px solid rgba(16,185,129,.3)">🟢 HOLD: {nc["HOLD"]}</span>'
            f'<span style="background:rgba(148,163,184,.1);color:#94a3b8;'
            f'padding:.3rem .8rem;border-radius:6px;font-size:.8rem;font-weight:800;'
            f'border:1px solid rgba(148,163,184,.3)">⚪ WATCH: {nc["WATCH"]}</span>'
            f'</div>',
            unsafe_allow_html=True)

        # ── 🎯 Exit Manager — R-multiples & professional exit rules ────────────
        if not odf.empty:
            _sig_by_stock = {s.get("stock"): s
                             for s in (st.session_state.signals_cache or [])}
            _em_rows = []
            for _, _pos in odf.iterrows():
                try:
                    _stk = _pos["stock"]
                    _buy = float(_pos["buy_at"])
                    _cmp_p = _pos.get("cmp")
                    if pd.isna(_cmp_p):
                        continue
                    _cmp_p = float(_cmp_p)
                    _sg_ = _sig_by_stock.get(_stk, {})
                    _stp = _sg_.get("stop_loss")
                    _pct = float(_pos.get("profit_pct", 0) or 0)
                    try:
                        _days = (datetime.now() -
                                 pd.to_datetime(_pos.get("added_date"))).days
                    except Exception:
                        _days = None
                    # R-multiple: profit measured in units of entry risk
                    _r_mult = None
                    if _stp and float(_stp) < _buy:
                        _risk0 = _buy - float(_stp)
                        if _risk0 > 0.01:
                            _r_mult = round((_cmp_p - _buy) / _risk0, 2)
                    # Exit rule ladder (first match wins)
                    if _stp and _cmp_p <= float(_stp):
                        _act = "🛑 EXIT — stop violated"
                    elif _pct <= -8:
                        _act = "⚠️ Review — beyond 8% risk cap"
                    elif _r_mult is not None and _r_mult >= 2:
                        _act = "📤 Book ⅓–½ profit, trail the rest"
                    elif _r_mult is not None and _r_mult >= 1:
                        _act = f"🛡 Move SL to breakeven (₹{_buy:,.2f})"
                    elif _days is not None and _days >= 15 and -2 <= _pct <= 2:
                        _act = "⏰ Time stop — dead money, redeploy"
                    elif _stp and float(_stp) >= _buy:
                        _act = "✅ Risk-free (stop ≥ entry) — trail"
                    else:
                        _act = "✋ Hold per plan"
                    _em_rows.append({
                        "Stock": _stk, "Entry": _buy, "CMP": _cmp_p,
                        "P&L %": _pct, "R": _r_mult,
                        "SL": float(_stp) if _stp else None,
                        "Days": _days, "Action": _act})
                except Exception:
                    continue
            if _em_rows:
                st.markdown('<div class="sec">🎯 Exit Manager — R-Multiples & Rules</div>',
                            unsafe_allow_html=True)
                _em_df = pd.DataFrame(_em_rows)
                _urgent = int(_em_df["Action"].str.contains("🛑|⚠️|⏰").sum())
                if _urgent:
                    st.warning(f"⚡ {_urgent} position(s) need action — see the Action column.",
                               icon="🎯")
                st.dataframe(
                    _em_df, hide_index=True, use_container_width=True,
                    column_config={
                        "Stock": st.column_config.TextColumn("Stock", width="small", pinned=True),
                        "Entry": st.column_config.NumberColumn("Entry", format="₹%.2f"),
                        "CMP":   st.column_config.NumberColumn("CMP", format="₹%.2f"),
                        "P&L %": st.column_config.NumberColumn("P&L %", format="%+.1f"),
                        "R":     st.column_config.NumberColumn("R", format="%.2f"),
                        "SL":    st.column_config.NumberColumn("SL", format="₹%.2f"),
                        "Days":  st.column_config.NumberColumn("Days", format="%d"),
                        "Action": st.column_config.TextColumn("Action", width="large"),
                    })
                st.caption("R = profit in units of entry risk (entry − stop). Playbook: "
                           "+1R → stop to breakeven · +2R → book ⅓–½ and trail the rest "
                           "(Supertrend) · flat after 15 days → time stop. Note: the SL "
                           "shown is the engine's CURRENT stop, so R is an approximation "
                           "once stops have trailed up.")

        render_signals(st.session_state.signals_cache, theme_t)

# ── Sector Rotation ──────────────────────────────────────────────────────────
elif _page == 'sector':
    st.markdown('<div class="sec">Macro Sector Rotation & Capital Flow</div>',
                unsafe_allow_html=True)

    if st.session_state.sector_cache is not None:
        render_sector(st.session_state.sector_cache, theme_t)

        # ── 🗺 Sector Heatmap (treemap) — where capital is flowing ─────────────
        try:
            _sh_df = st.session_state.sector_cache.copy()
            if not _sh_df.empty and "avg_pct" in _sh_df.columns:
                _sh_df["_w"] = (_sh_df["momentum_score"].clip(lower=0.05)
                                if "momentum_score" in _sh_df.columns else 1.0)
                _tfig = px.treemap(
                    _sh_df, path=["sector"], values="_w", color="avg_pct",
                    color_continuous_scale=["#ef4444", "#3f3f46", "#10b981"],
                    color_continuous_midpoint=0, custom_data=["avg_pct"])
                _tfig.update_traces(
                    texttemplate="<b>%{label}</b><br>%{customdata[0]:+.1f}%",
                    textfont=dict(size=13))
                _tfig.update_layout(
                    height=340, margin=dict(l=4, r=4, t=8, b=4),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=theme_t.get("text", "#fff")),
                    coloraxis_showscale=False)
                st.plotly_chart(_tfig, use_container_width=True)
                st.caption("Tile size = sector momentum · colour = 1-month flow "
                           "(green = money in, red = money out). One glance shows "
                           "where capital is rotating.")
        except Exception:
            pass

        if not st.session_state.sector_cache.empty:
            top = st.session_state.sector_cache.iloc[0]
            rs_val  = top.get("rs_vs_nifty_1m", 0) or 0
            rs_clr  = "#10b981" if rs_val > 0 else "#ef4444"
            rrg_val = top.get("rrg_quadrant", "—")

            st.markdown(
                f'<div style="margin-top:1rem;background:rgba(16,185,129,.08);'
                f'border:1px solid rgba(16,185,129,.3);border-radius:8px;'
                f'padding:.8rem 1rem;font-size:.85rem">'
                f'🥇 <b style="color:var(--text)">Leading Sector: {top["sector"]}</b>'
                f' — Momentum {top["momentum_score"]:.2f}'
                f' | Avg RSI {top["avg_rsi"]:.0f}'
                f' | Flow {top["avg_pct"]:+.1f}%'
                f' | RS vs Nifty <b style="color:{rs_clr}">{rs_val:+.1f}%</b>'
                f' | {rrg_val}<br>'
                f'<span style="color:var(--muted);font-size:.75rem;margin-top:5px;'
                f'display:block">Constituents: {top["stocks"]}</span>'
                f'</div>',
                unsafe_allow_html=True)

    if (st.session_state.outlook_cache is not None and
            not st.session_state.outlook_cache.empty):
        st.markdown(
            '<div class="sec" style="margin-top:2rem">📈 Institutional Outlook</div>',
            unsafe_allow_html=True)
        render_outlook(st.session_state.outlook_cache, theme_t)

    if st.session_state.picks_cache is not None:
        st.markdown(
            '<div class="sec" style="margin-top:2rem">🎯 Algorithmic Entry Setups</div>',
            unsafe_allow_html=True)
        render_picks(st.session_state.picks_cache, theme_t)

# ── Theme Scanner ────────────────────────────────────────────────────────────
elif _page == 'themes':
    st.markdown('<div class="sec">🎯 Market Theme Scanner</div>',
                unsafe_allow_html=True)
    st.caption("Curated NSE market themes (Defence, Railways, EV, PSU Banks, "
               "Green Energy, Capex…) ranked by aggregate strength — momentum, "
               "breadth above key EMAs, and relative strength vs Nifty. See where "
               "capital is rotating, then drill into each theme's leaders.")

    if not _THEMES_AVAILABLE:
        st.warning("🎯 Theme Scanner needs `theme_scanner.py` in your repo root "
                   "(same folder as app.py and signals.py). Upload it, then reboot.",
                   icon="⚠️")
    else:
        tc1, tc2 = st.columns([3, 1])
        with tc1:
            st.markdown(
                '<div style="font-size:.8rem;color:var(--muted);font-weight:600">'
                'Each theme is a curated basket drawn from your existing universe. '
                'Edit the lists in <code>theme_scanner.py</code> any time.</div>',
                unsafe_allow_html=True)
        with tc2:
            run_theme = st.button("🎯 Scan Themes", width="stretch")

        with st.expander("🔍 Theme basket coverage", expanded=False):
            try:
                st.code(_themes.theme_coverage(), language=None)
            except Exception as _e:
                st.caption(f"Coverage unavailable: {_e}")

        if run_theme:
            with st.spinner("Scoring NSE themes by aggregate strength…"):
                st.session_state.theme_scan_cache = _themes.scan_themes(
                    min_constituents=3)

        tdata = st.session_state.get("theme_scan_cache")
        if tdata is None:
            st.info("💡 Click **🎯 Scan Themes** to rank NSE themes hottest → coldest.")
        elif not tdata.get("themes"):
            st.warning("No themes had enough scored constituents. "
                       + tdata.get("note", "Try again — Yahoo may be rate-limited."))
        else:
            themes = tdata["themes"]
            st.markdown(
                f'<div style="font-size:.75rem;color:var(--muted);margin-bottom:1rem">'
                f'Ranked {len(themes)} themes · scanned {tdata["scanned"]} stocks · '
                f'{tdata["timestamp"]}</div>',
                unsafe_allow_html=True)

            cards = ""
            for t in themes:
                rank = t["rank"]
                score = t["score"]
                medal = ("🥇" if rank == 1 else "🥈" if rank == 2 else
                         "🥉" if rank == 3 else f"#{rank}")
                if score >= 55:
                    clr = theme_t["green"]; bdr = theme_t["accent"]
                elif score >= 30:
                    clr = theme_t["yellow"]; bdr = theme_t["border"]
                else:
                    clr = theme_t["muted"]; bdr = theme_t["border"]
                rs = t["avg_rs"]
                rs_clr = theme_t["green"] if rs >= 1.0 else theme_t["red"]
                cards += (
                    f'<div style="background:var(--card);border:1px solid {bdr};'
                    f'border-radius:12px;padding:1rem 1.1rem;min-width:200px;flex:1">'
                    f'<div style="display:flex;justify-content:space-between;'
                    f'align-items:center;margin-bottom:.3rem">'
                    f'<span style="font-size:.82rem;font-weight:800;color:var(--text)">'
                    f'{medal} {t["theme"]}</span></div>'
                    f'<div style="font-size:1.6rem;font-weight:800;color:{clr};'
                    f'margin:.1rem 0">{score:.0f}'
                    f'<span style="font-size:.6rem;color:var(--muted)"> /100</span></div>'
                    f'<div style="height:5px;background:var(--input);border-radius:3px;'
                    f'margin:.3rem 0 .5rem"><div style="height:5px;border-radius:3px;'
                    f'background:{clr};width:{min(score,100):.0f}%"></div></div>'
                    f'<div style="font-size:.68rem;color:var(--muted);line-height:1.5">'
                    f'1M <b style="color:var(--text)">{t["avg_ret_1m"]:+.1f}%</b> · '
                    f'3M <b style="color:var(--text)">{t["avg_ret_3m"]:+.1f}%</b><br>'
                    f'{t["pct_above_50ema"]:.0f}% &gt;50EMA · '
                    f'RS <b style="color:{rs_clr}">{rs:.2f}</b> · '
                    f'{t["n_scored"]} stks</div></div>'
                )
            st.markdown(
                f'<div style="display:flex;gap:.6rem;flex-wrap:wrap;'
                f'margin-bottom:1.5rem">{cards}</div>',
                unsafe_allow_html=True)

            st.markdown('<div class="sec">🔬 Theme Constituents</div>',
                        unsafe_allow_html=True)
            for t in themes:
                medal = ("🥇" if t["rank"] == 1 else "🥈" if t["rank"] == 2 else
                         "🥉" if t["rank"] == 3 else f"#{t['rank']}")
                with st.expander(
                        f"{medal} {t['theme']} — score {t['score']:.0f} · "
                        f"1M {t['avg_ret_1m']:+.1f}% · breadth {t['breadth_up']:.0f}% up · "
                        f"{t['n_scored']} stocks", expanded=(t["rank"] == 1)):
                    rows = []
                    for l in t["leaders"]:
                        rows.append({
                            "Stock": l["stock"],
                            "CMP": l["cmp"],
                            "1M %": l["ret_1m"],
                            "3M %": l["ret_3m"],
                            "RS": l["rs_ratio"],
                            ">50EMA": "✅" if l["above_50ema"] else "—",
                            ">200EMA": "✅" if l["above_200ema"] else "—",
                            "Trend": l["trend"],
                        })
                    import pandas as _pd_t
                    tdf = _pd_t.DataFrame(rows)
                    _h = min(max(len(tdf) * 36 + 40, 150), 480)
                    st.dataframe(
                        tdf, hide_index=True, height=_h, use_container_width=True,
                        column_config={
                            "Stock": st.column_config.TextColumn("Stock", width="small", pinned=True),
                            "CMP":   st.column_config.NumberColumn("CMP", format="₹%.2f"),
                            "1M %":  st.column_config.NumberColumn("1M %", format="%.1f"),
                            "3M %":  st.column_config.NumberColumn("3M %", format="%.1f"),
                            "RS":    st.column_config.NumberColumn("RS", format="%.2f"),
                        })
                    st.download_button(
                        f"⬇️ Export {t['theme']} CSV",
                        tdf.to_csv(index=False).encode("utf-8"),
                        file_name=f"theme_{t['theme'].replace(' ','_').replace('/','-')}_"
                                  f"{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv", key=f"theme_dl_{t['rank']}")
            st.caption("💡 Hot themes (green, score 55+) show where capital is rotating. "
                       "Inside each, stocks are ranked by relative strength — the leaders "
                       "of the leading themes are your highest-conviction shortlist. "
                       "Always confirm your own entry setup before trading.")


# ── Universe Scanner ─────────────────────────────────────────────────────────
elif _page == 'scanner':
    st.markdown(
        f'<div class="sec">🌌 Universe Scanner — {UNIVERSE_TOTAL:,} Assets</div>',
        unsafe_allow_html=True)

    # ── Universe source breakdown ─────────────────────────────────────────────
    src_html = ""
    for lbl, n, sk, err in UNIVERSE_SOURCES:
        clr = theme_t["green"] if n > 0 else theme_t["red"]
        src_html += (
            f'<span style="background:var(--card2);border:1px solid var(--border);'
            f'border-radius:6px;padding:.25rem .6rem;font-size:.72rem;'
            f'font-weight:700;color:var(--text)">'
            f'{"📄" if n > 0 else "❌"} {lbl} '
            f'<span style="color:{clr}">{n:,}</span></span> '
        )
    st.markdown(
        f'<div style="display:flex;gap:.4rem;flex-wrap:wrap;margin-bottom:.8rem;'
        f'align-items:center">'
        f'<span style="font-size:.75rem;color:var(--muted);font-weight:600">'
        f'Sources loaded:</span> {src_html}</div>',
        unsafe_allow_html=True)

    # ── Diagnostic expander — shows exactly what loaded and why ───────────────
    with st.expander("🔍 Universe Load Diagnostics", expanded=False):
        st.code(debug_universe_load(), language=None)
        st.caption("If a file shows '❌ not found', check it is committed to "
                   "your repo root (same folder as signals.py and app.py).")

    st.markdown(
        f'<div style="background:rgba(212,175,55,.06);border:1px solid rgba(212,175,55,.2);'
        f'border-radius:8px;padding:.7rem 1rem;font-size:.8rem;color:var(--muted);'
        f'margin-bottom:1rem;line-height:1.6">'
        f'💡 <b style="color:var(--text)">Add more stocks:</b> Download CSVs from '
        f'<a href="https://www.nseindia.com" target="_blank" '
        f'style="color:var(--accent)">nseindia.com</a> and commit to your repo root.<br>'
        f'Supported: <code>ind_nifty1000list.csv</code> · '
        f'<code>ind_niftymidcap150list.csv</code> · '
        f'<code>ind_niftysmallcap250list.csv</code> · '
        f'<code>ind_niftymicrocap250list.csv</code> · '
        f'<code>EQUITY_L.csv</code> (all ~2,000 NSE equities)'
        f'</div>',
        unsafe_allow_html=True)

    # ── Custom stock input ─────────────────────────────────────────────────────
    with st.expander("➕ Add Custom Stocks to Scan", expanded=False):
        st.caption("Enter NSE symbols (comma-separated). These are added to the scan universe temporarily.")
        custom_raw = st.text_area(
            "Custom symbols", value=st.session_state.get("custom_stocks_input",""),
            placeholder="IRFC, CDSL, SNOWMAN, ZOMATO...",
            label_visibility="collapsed", height=80)
        if st.button("✅ Apply Custom List"):
            # Store and inject into signals module
            symbols = [s.strip().upper() for s in custom_raw.split(",") if s.strip()]
            st.session_state.custom_stocks_input = custom_raw
            if symbols:
                import signals as _sg
                if "Custom" not in _sg.SECTOR_STOCKS:
                    _sg.SECTOR_STOCKS["Custom"] = []
                _sg.SECTOR_STOCKS["Custom"] = symbols
                for sym in symbols:
                    _sg.SECTOR_MAP[sym] = "Custom"
                st.success(f"✅ Added {len(symbols)} custom stocks to scan universe")
            else:
                import signals as _sg
                _sg.SECTOR_STOCKS.pop("Custom", None)
                st.info("Custom list cleared.")

    if st.button("⚡ Execute Global Scan", width="stretch"):
        with st.spinner(f"Scanning {len(SECTOR_MAP)} tickers..."):
            sd = generate_market_scanner()
            st.session_state.scanner_cache = sd if (sd is not None and not sd.empty) \
                else pd.DataFrame()

            # ── Scanner diagnostics (works once signals.py's get_scanner_
            #    diagnostics() patch is applied; degrades silently otherwise) ──
            try:
                import signals as _sg_diag
                if hasattr(_sg_diag, "get_scanner_diagnostics"):
                    st.session_state["_scanner_diag"] = _sg_diag.get_scanner_diagnostics()
            except Exception:
                pass

            if (st.session_state.scanner_cache is not None and
                    not st.session_state.scanner_cache.empty):
                _sc = st.session_state.scanner_cache
                _has_liq = "Liquid" in _sc.columns
                liq_ok  = int((_sc["Liquid"]=="✅").sum()) if _has_liq else 0
                liq_low = int((_sc["Liquid"]=="⚠️ Low").sum()) if _has_liq else 0
                st.toast(
                    f"✅ {len(_sc)} setups | "
                    f"💧 {liq_ok} liquid · ⚠️ {liq_low} low-liq",
                    icon="🚀")

            # ── AUTO-CHAIN Scanner 2.0 right after Universe Scan ────────────────
            # Runs the regime/RS/structural-gate rework on the same universe and
            # stores a funnel comparison: how many Universe Scan found vs how
            # many survive Scanner 2.0's stricter filters (shown below).
            try:
                import scanner_v2 as _sv2_auto
                _r2_auto = _sv2_auto.generate_market_scanner_v2()
                st.session_state.scan2_cache = _r2_auto
                _d2_auto = _r2_auto.get("df")
                _v1_count = len(st.session_state.scanner_cache) \
                    if st.session_state.scanner_cache is not None else 0
                _v2_count = len(_d2_auto) if _d2_auto is not None else 0
                _v2_buy = (int(_d2_auto["Signal"].isin(
                            ["🔥 STRONG BUY", "🟢 BUY SETUP"]).sum())
                          if _d2_auto is not None and not _d2_auto.empty else 0)
                st.session_state["_scan_funnel"] = {
                    "v1_total": _v1_count, "v2_scored": _v2_count,
                    "v2_buy": _v2_buy, "regime": _r2_auto.get("regime", "—"),
                }
            except Exception as _sv2_err:
                st.session_state["_scan_funnel"] = None
                st.session_state["_scan_funnel_error"] = str(_sv2_err)

            st.rerun()

    # ── Persistent funnel card: Universe Scan → Scanner 2.0 conversion ────────
    _funnel = st.session_state.get("_scan_funnel")
    if _funnel:
        st.markdown(
            f'<div style="background:var(--card);border:1px solid var(--accent);'
            f'border-radius:10px;padding:.8rem 1.1rem;margin-bottom:1rem;'
            f'display:flex;gap:1.5rem;align-items:center;flex-wrap:wrap">'
            f'<span style="font-weight:800;color:var(--accent)">🚀 Scanner 2.0 '
            f'auto-check</span>'
            f'<span style="font-size:.85rem;color:var(--text)">'
            f'Universe Scan found <b>{_funnel["v1_total"]}</b> → '
            f'Scanner 2.0 scored <b>{_funnel["v2_scored"]}</b> → '
            f'<b style="color:var(--green)">{_funnel["v2_buy"]}</b> pass as '
            f'BUY-tier (regime: {_funnel["regime"]})</span>'
            f'<span style="font-size:.75rem;color:var(--muted)">'
            f'See full breakdown on 🚀 Scanner 2.0 page</span>'
            f'</div>', unsafe_allow_html=True)
    elif st.session_state.get("_scan_funnel_error"):
        st.caption(f"🚀 Scanner 2.0 auto-check skipped — {st.session_state['_scan_funnel_error']} "
                   "(upload scanner_v2.py to enable this comparison)")

    # ── Scanner diagnostics expander (why results are sparse, if ever) ───────
    _diag_text = st.session_state.get("_scanner_diag")
    if _diag_text:
        with st.expander("🔍 Why this many results? (fetch/liquidity breakdown)"):
            st.code(_diag_text, language=None)

    scan_df = st.session_state.scanner_cache
    if scan_df is None:
        st.info("💡 Initiate scan above or await automated background scan.")
    elif scan_df.empty:
        st.warning("⚠️ Zero setups passed pattern gates today.")
    else:
        all_sectors = sorted(scan_df["Sector"].unique().tolist())

        # ── Sector summary cards ───────────────────────────────────────────────
        _has_liq_col = "Liquid" in scan_df.columns
        _agg = {"total": ("Stock","count"),
                "strong": ("Signal", lambda x: (x=="🔥 STRONG BUY").sum()),
                "buy":    ("Signal", lambda x: (x=="🟢 BUY SETUP").sum())}
        if _has_liq_col:
            _agg["liq"] = ("Liquid", lambda x: (x=="✅").sum())
        sector_stats = scan_df.groupby("Sector").agg(**_agg).reset_index()
        if not _has_liq_col:
            sector_stats["liq"] = 0
        cards_html = ""
        for _, sr in sector_stats.iterrows():
            hot = sr["strong"] + sr["buy"]
            clr = theme_t["green"] if hot > 0 else theme_t["muted"]
            bdr = theme_t["accent"] if hot > 0 else theme_t["border"]
            cards_html += (
                f'<div style="background:var(--card);border:1px solid {bdr};'
                f'border-radius:10px;padding:.8rem 1rem;min-width:150px;flex:1">'
                f'<div style="font-size:.78rem;font-weight:800;color:var(--text);'
                f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'
                f'{sr["Sector"]}</div>'
                f'<div style="font-size:1.1rem;font-weight:800;color:{clr};margin:.2rem 0">'
                f'{int(sr["total"])}<span style="font-size:.65rem;color:var(--muted)"> stocks</span></div>'
                f'<div style="font-size:.68rem;color:var(--muted)">'
                f'🔥{int(sr["strong"])} 🟢{int(sr["buy"])} '
                f'· 💧{int(sr["liq"])} liquid</div></div>'
            )
        st.markdown(
            f'<div style="display:flex;gap:.5rem;flex-wrap:wrap;'
            f'margin-bottom:1rem">{cards_html}</div>',
            unsafe_allow_html=True)

        # ── Stable filter controls ──────────────────────────────────────────────
        fc1, fc2, fc3, fc4, fc5 = st.columns([2, 1.5, 1.2, 1, 1])
        with fc1:
            sector_options = ["All Sectors"] + all_sectors
            sel_sector = st.selectbox("Sector", sector_options,
                index=sector_options.index(st.session_state.selected_scanner_sector)
                if st.session_state.selected_scanner_sector in sector_options else 0,
                label_visibility="collapsed")
            if sel_sector != st.session_state.selected_scanner_sector:
                st.session_state.selected_scanner_sector = sel_sector
        with fc2:
            signal_opts = ["All Signals","🔥 STRONG BUY","🟢 BUY SETUP",
                           "🟡 ACCUMULATE","⚪ NEUTRAL","🔴 AVOID"]
            sel_signal = st.selectbox("Signal", signal_opts, label_visibility="collapsed")
        with fc3:
            _has_liq_col2 = "Liquid" in scan_df.columns if scan_df is not None else False
            liq_opts = (["All","✅ Liquid Only","⚠️ Low Liq Only"]
                        if _has_liq_col2 else ["All"])
            sel_liq = st.selectbox("Liquidity", liq_opts, label_visibility="collapsed")
        with fc4:
            search_stock = st.text_input("Search", placeholder="Symbol",
                                         label_visibility="collapsed")
        with fc5:
            min_score_f = st.number_input("Min score", 0, 10, 0, 1,
                                          label_visibility="collapsed")

        # ── Apply filters ───────────────────────────────────────────────────────
        fdf = scan_df.copy()
        if sel_sector != "All Sectors":
            fdf = fdf[fdf["Sector"] == sel_sector]
        if sel_signal != "All Signals":
            fdf = fdf[fdf["Signal"] == sel_signal]
        if "Liquid" in fdf.columns:
            if sel_liq == "✅ Liquid Only":
                fdf = fdf[fdf["Liquid"] == "✅"]
            elif sel_liq == "⚠️ Low Liq Only":
                fdf = fdf[fdf["Liquid"] == "⚠️ Low"]
        if search_stock.strip():
            fdf = fdf[fdf["Stock"].str.upper().str.contains(
                search_stock.strip().upper())]
        if min_score_f > 0:
            fdf = fdf[fdf["Score"] >= min_score_f]

        # ── Strategy quick-filters (VCP / RS leaders / hide traps) ──────────────
        if any(col in fdf.columns for col in ("VCP", "RS", "Trap")):
            qf1, qf2, qf3, qf4 = st.columns(4)
            with qf1:
                only_vcp = st.checkbox("📐 VCP bases only", value=False, key="scn_vcp")
            with qf2:
                only_rs = st.checkbox("💪 RS leaders only", value=False, key="scn_rs")
            with qf3:
                ready_only = st.checkbox("🎯 VCP pivot-ready", value=False, key="scn_ready")
            with qf4:
                hide_traps = st.checkbox("🚫 Hide traps", value=False, key="scn_notrap")
            if only_vcp and "VCP" in fdf.columns:
                fdf = fdf[fdf["VCP"] != "—"]
            if ready_only and "VCP" in fdf.columns:
                fdf = fdf[fdf["VCP"].str.contains("READY", na=False)]
            if only_rs and "RS_Lead" in fdf.columns:
                fdf = fdf[fdf["RS_Lead"] == "💪"]
            if hide_traps and "Trap" in fdf.columns:
                fdf = fdf[fdf["Trap"] == "—"]

        display_df = fdf.drop(columns=["Sector"]) if sel_sector != "All Sectors" else fdf

        liq_count = int((fdf["Liquid"]=="✅").sum()) if "Liquid" in fdf.columns else 0
        st.markdown(
            f'<div style="font-size:.78rem;color:var(--muted);margin-bottom:.4rem;'
            f'font-weight:600">Showing {len(fdf)} of {len(scan_df)} results · '
            f'💧 {liq_count} liquid</div>',
            unsafe_allow_html=True)

        # ── Single stable dataframe with fixed height ───────────────────────────
        st.dataframe(
            display_df.reset_index(drop=True),
            hide_index=True, height=600, use_container_width=True,
            column_config={
                "Generated":    st.column_config.TextColumn("Time",     width="small"),
                "Sector":       st.column_config.TextColumn("Sector",   width="medium"),
                "Stock":        st.column_config.TextColumn("Stock",    width="small"),
                "Signal":       st.column_config.TextColumn("Signal",   width="medium"),
                "Liquid":       st.column_config.TextColumn("💧 Liq",   width="small"),
                "Turnover_Cr":  st.column_config.NumberColumn("₹Cr/day",format="%.1f"),
                "Score":        st.column_config.NumberColumn("Score",  format="%d"),
                "CMP":     st.column_config.NumberColumn("CMP",    format="₹%.2f"),
                "Entry":   st.column_config.NumberColumn("Entry",  format="₹%.2f"),
                "Target":  st.column_config.NumberColumn("Target", format="₹%.2f"),
                "SL":      st.column_config.NumberColumn("SL",     format="₹%.2f"),
                "Support": st.column_config.NumberColumn("Support",format="₹%.2f"),
                "Resist":  st.column_config.NumberColumn("Resist", format="₹%.2f"),
                "RSI":     st.column_config.NumberColumn("RSI",    format="%.1f"),
                "Trend":   st.column_config.TextColumn("Trend",   width="medium"),
                "VCP":     st.column_config.TextColumn("📐 VCP",  width="small"),
                "Trap":    st.column_config.TextColumn("🪤 Trap", width="medium"),
                "RS":      st.column_config.NumberColumn("💪 RS", format="%.2f"),
                "RS_Lead": st.column_config.TextColumn("Lead",    width="small"),
                "Patterns":st.column_config.TextColumn("Patterns",width="large"),
            })

# ── Scanner 2.0 (regime-aware, RS-gated, structural stops) ──────────────────
elif _page == 'scanner2':
    st.markdown('<div class="sec">🚀 Universe Scanner 2.0</div>',
                unsafe_allow_html=True)
    st.caption("Audit-grade rework: regime gating · RS gate · extension penalty · "
               "fresh-breakout-only credit · structural stops · measured-move targets "
               "· percentile ranking · outcome logging. Your original scanner is "
               "untouched — compare them side by side.")

    try:
        import scanner_v2 as _sv2
        _SV2_OK = True
    except Exception as _sv2e:
        _SV2_OK = False
        st.warning(f"🚀 Scanner 2.0 needs `scanner_v2.py` in the repo root "
                   f"(same folder as app.py). Upload it, then reboot. ({_sv2e})",
                   icon="⚠️")

    if _SV2_OK:
        if "scan2_cache" not in st.session_state:
            st.session_state.scan2_cache = None

        # ── Portfolio heat gauge (open risk vs capital, from live signals) ────
        _heat_rs = 0.0
        try:
            for _s in (st.session_state.signals_cache or []):
                _c, _sl, _qty = _s.get("cmp"), _s.get("stop_loss"), _s.get("quantity")
                if _c and _sl and _qty and _c > _sl:
                    _heat_rs += float(_qty) * (float(_c) - float(_sl))
        except Exception:
            pass
        _cap = float(st.session_state.get("_sz_cap", 100000.0))
        _heat_pct = (_heat_rs / _cap * 100) if _cap > 0 else 0
        _h_clr = "#10b981" if _heat_pct < 4 else "#f59e0b" if _heat_pct < 6 else "#ef4444"
        _h_msg = ("OK — room for new entries" if _heat_pct < 4 else
                  "Caution — near risk budget" if _heat_pct < 6 else
                  "🔴 OVER BUDGET — avoid new entries until risk reduces")
        st.markdown(
            f'<div style="background:var(--card);border:1px solid {_h_clr};'
            f'border-radius:10px;padding:.7rem 1.1rem;margin-bottom:1rem;'
            f'display:flex;gap:1.2rem;align-items:center;flex-wrap:wrap">'
            f'<span style="font-weight:800;color:{_h_clr}">🌡 Portfolio Heat: '
            f'{_heat_pct:.1f}%</span>'
            f'<span style="font-size:.78rem;color:var(--muted)">open risk '
            f'₹{_heat_rs:,.0f} of ₹{_cap:,.0f} capital · {_h_msg}</span></div>',
            unsafe_allow_html=True)

        sc2a, sc2b = st.columns([3, 1])
        with sc2b:
            _run_v2 = st.button("🚀 Run Scan 2.0", width="stretch")
        with sc2a:
            st.markdown('<div style="font-size:.78rem;color:var(--muted);'
                        'font-weight:600;padding-top:.6rem">⚡ = fresh breakout '
                        '(last 2 bars) · Entry = pivot buy-stop for fresh setups '
                        '· SL below swing low · RR from actionable entry</div>',
                        unsafe_allow_html=True)

        if _run_v2:
            with st.spinner("Running regime-aware scan…"):
                st.session_state.scan2_cache = _sv2.generate_market_scanner_v2()

        # ── 🔍 Check any specific stock (doesn't need to be in the curated
        #    universe — works for any NSE symbol signals.py can fetch) ────────
        st.markdown('<div style="font-size:.85rem;font-weight:800;margin:1rem 0 .4rem">'
                    '🔍 Check a Specific Stock</div>', unsafe_allow_html=True)
        _lk1, _lk2 = st.columns([3, 1])
        with _lk1:
            _lookup_sym = st.text_input("Symbol", key="s2_lookup_sym",
                                        placeholder="e.g. RELIANCE, TCS, IRFC…",
                                        label_visibility="collapsed")
        with _lk2:
            _lookup_go = st.button("🔍 Check", width="stretch", key="s2_lookup_btn")

        if _lookup_go and _lookup_sym.strip():
            with st.spinner(f"Scoring {_lookup_sym.strip().upper()}…"):
                _lk_result = _sv2.score_single_stock(_lookup_sym.strip())
            if "error" in _lk_result:
                st.error(f"⚠️ {_lk_result['error']}")
            else:
                _lk = _lk_result
                _lk_clr = ("#10b981" if "STRONG BUY" in _lk["Signal"] or "BUY SETUP" in _lk["Signal"]
                          else "#ef4444" if "AVOID" in _lk["Signal"] or "RISK WIDE" in _lk["Signal"]
                          else "#f59e0b" if "WATCH" in _lk["Signal"] or "ACCUMULATE" in _lk["Signal"]
                          else "#8e8e93")
                st.markdown(
                    f'<div style="background:var(--card);border:1px solid {_lk_clr};'
                    f'border-radius:12px;padding:1.1rem 1.3rem;margin:.6rem 0">'
                    f'<div style="display:flex;justify-content:space-between;'
                    f'align-items:center;margin-bottom:.6rem">'
                    f'<span style="font-size:1.1rem;font-weight:800">{_lk["Stock"]} '
                    f'<span style="font-size:.75rem;color:var(--muted);font-weight:400">'
                    f'{_lk["Sector"]}</span></span>'
                    f'<span style="background:{_lk_clr}22;color:{_lk_clr};'
                    f'padding:.25rem .7rem;border-radius:6px;font-size:.8rem;'
                    f'font-weight:800">{_lk["Signal"]}</span></div>'
                    f'<div style="display:grid;grid-template-columns:repeat(4,1fr);'
                    f'gap:.6rem;font-size:.82rem">'
                    f'<div><span style="color:var(--muted)">Score</span><br>'
                    f'<b>{_lk["Score"]}</b></div>'
                    f'<div><span style="color:var(--muted)">CMP</span><br>'
                    f'<b>₹{_lk["CMP"]}</b></div>'
                    f'<div><span style="color:var(--muted)">Entry</span><br>'
                    f'<b>₹{_lk["Entry"]}</b></div>'
                    f'<div><span style="color:var(--muted)">SL</span><br>'
                    f'<b>₹{_lk["SL"]}</b> <span style="color:var(--muted);'
                    f'font-size:.72rem">({_lk["SL basis"]})</span></div>'
                    f'<div><span style="color:var(--muted)">Target</span><br>'
                    f'<b>₹{_lk["Target"]}</b></div>'
                    f'<div><span style="color:var(--muted)">R:R</span><br>'
                    f'<b>{_lk["RR"] if _lk["RR"] is not None else "—"}</b></div>'
                    f'<div><span style="color:var(--muted)">RS</span><br>'
                    f'<b>{_lk["RS"] if _lk["RS"] is not None else "—"}</b></div>'
                    f'<div><span style="color:var(--muted)">Fresh</span><br>'
                    f'<b>{_lk["Fresh"] or "—"}</b></div>'
                    f'</div>'
                    f'<div style="font-size:.72rem;color:var(--muted);margin-top:.6rem">'
                    f'Regime: {_lk["regime"]} · Trend: {_lk["Trend"]} · RSI {_lk["RSI"]} · '
                    f'{_lk["timestamp"]}</div>'
                    f'</div>', unsafe_allow_html=True)
                st.caption("⚠️ Single-stock lookups use absolute score thresholds "
                          "(no percentile rank — nothing to rank against with just "
                          "one stock). Run the full scan for percentile-ranked "
                          "STRONG BUY / BUY SETUP tiers.")


        _r2 = st.session_state.scan2_cache
        if _r2 is not None:
            _stale_banner(_r2.get("timestamp"), "🚀 Run Scan 2.0")
        if _r2 is None:
            st.info("💡 Click **Run Scan 2.0**. First run fetches history "
                    "(shares the same cache as your other scanners).")
        elif _r2["df"].empty:
            st.warning("Scan returned nothing — see the breakdown below to find out why.")
            _dg = _r2.get("diagnostics") or {}
            if _dg:
                st.markdown(
                    f"**🔍 Scanner 2.0 breakdown:**\n"
                    f"- Universe attempted: **{_dg.get('total', 0):,}**\n"
                    f"- ❌ No data fetched (Yahoo/Angel empty): **{_dg.get('no_data', 0):,}**\n"
                    f"- ❌ Indicators couldn't compute: **{_dg.get('no_ind', 0):,}**\n"
                    f"- ⚠️ Fetched but couldn't be scored: **{_dg.get('score_none', 0):,}**\n"
                    f"- 💥 Exceptions during scoring: **{_dg.get('exc', 0):,}**")
                if _dg.get("last_exc"):
                    st.caption(f"Last exception: `{_dg['last_exc']}`")
                if _dg.get("no_data", 0) > _dg.get("total", 1) * 0.5:
                    st.info("Most symbols returned no price data — the data source "
                            "(Angel/Yahoo) is likely rate-limited or unreachable right "
                            "now. Wait a few minutes and retry, or check the data-source "
                            "logs.")
        else:
            _d2 = _r2["df"]
            _d2 = _d2.copy()
            # Heat is shown informationally in the gauge above (no signals are
            # blocked or relabeled here) — you decide what to act on.
            # ── 📆 Earnings-soon flag (uses your Earnings Calendar fetch) ──────
            try:
                _ear_soon = {r.get("Stock") for r in
                             (st.session_state.get("_earnings_cache") or [])
                             if isinstance(r, dict) and r.get("Days Away", 99) <= 7}
            except Exception:
                _ear_soon = set()
            if _ear_soon:
                _d2.insert(3, "📆", _d2["Stock"].map(
                    lambda s: "📆" if s in _ear_soon else ""))
            _n_strong = int((_d2["Signal"] == "🔥 STRONG BUY").sum())
            _n_buy    = int((_d2["Signal"] == "🟢 BUY SETUP").sum())
            st.markdown(
                f'<div style="font-size:.78rem;color:var(--muted);margin:.3rem 0 .6rem;'
                f'font-weight:600">Regime: <b style="color:var(--text)">'
                f'{_r2["regime"]}</b> · {_r2["scanned"]} scored · '
                f'🔥 {_n_strong} strong · 🟢 {_n_buy} buy setups · '
                f'{_r2["timestamp"]}</div>', unsafe_allow_html=True)

            # ── Result filters ─────────────────────────────────────────────────
            _f1, _f2, _f3, _f4 = st.columns([1.4, 1.2, 1.2, 1.4])
            with _f1:
                _sig_opts = sorted(_d2["Signal"].unique().tolist())
                _sel_sig = st.multiselect("Signal", _sig_opts, default=[],
                                          key="s2_sig_filter",
                                          placeholder="All signals")
            with _f2:
                _min_score2 = st.number_input("Min score", value=-999.0, step=1.0,
                                              key="s2_min_score")
            with _f3:
                _sec_opts = sorted(_d2["Sector"].unique().tolist())
                _sel_sec = st.multiselect("Sector", _sec_opts, default=[],
                                         key="s2_sec_filter",
                                         placeholder="All sectors")
            with _f4:
                _search2 = st.text_input("Search stock", key="s2_search",
                                         placeholder="e.g. RELIANCE")

            _d2_view = _d2.copy()
            if _sel_sig:
                _d2_view = _d2_view[_d2_view["Signal"].isin(_sel_sig)]
            if _min_score2 > -999:
                _d2_view = _d2_view[_d2_view["Score"] >= _min_score2]
            if _sel_sec:
                _d2_view = _d2_view[_d2_view["Sector"].isin(_sel_sec)]
            if _search2.strip():
                _d2_view = _d2_view[_d2_view["Stock"].str.upper()
                                    .str.contains(_search2.strip().upper())]
            st.caption(f"Showing {len(_d2_view)} of {len(_d2)} results")

            _h2 = min(max(len(_d2_view) * 36 + 40, 240), 620)
            st.dataframe(
                _d2_view, hide_index=True, height=_h2, use_container_width=True,
                column_config={
                    "Stock":   st.column_config.TextColumn("Stock", width="small", pinned=True),
                    "Score":   st.column_config.NumberColumn("Score", format="%.1f"),
                    "Pctl":    st.column_config.ProgressColumn("Pctl", min_value=0, max_value=100, format="%.0f"),
                    "CMP":     st.column_config.NumberColumn("CMP", format="₹%.2f"),
                    "Entry":   st.column_config.NumberColumn("Entry", format="₹%.2f"),
                    "SL":      st.column_config.NumberColumn("SL", format="₹%.2f"),
                    "Target":  st.column_config.NumberColumn("Target", format="₹%.2f"),
                    "Risk %":  st.column_config.NumberColumn("Risk %", format="%.1f"),
                    "RR":      st.column_config.NumberColumn("R:R", format="%.2f"),
                    "RS":      st.column_config.NumberColumn("RS", format="%.2f"),
                    "RSI":     st.column_config.NumberColumn("RSI", format="%.0f"),
                    "Ext(ATR)": st.column_config.NumberColumn("Ext", format="%.1f"),
                })

            # ── Outcome logger — the audit's #1 missing feature ───────────────
            lg1, lg2 = st.columns([1, 2])
            with lg1:
                if st.button("📸 Log Top Signals", width="stretch",
                             help="Snapshot today's 🔥/🟢 signals to track forward returns"):
                    try:
                        db("CREATE TABLE IF NOT EXISTS signal_log("
                           "user_id INTEGER, log_date TEXT, scanner TEXT, "
                           "stock TEXT, score REAL, entry REAL, "
                           "stop_loss REAL, target REAL, rr REAL)")
                        _today_lg = datetime.now().strftime("%Y-%m-%d")
                        db("DELETE FROM signal_log WHERE user_id=? AND "
                           "log_date=? AND scanner=?", (UID, _today_lg, "v2"))
                        _top_lg = _d2[_d2["Signal"].isin(
                            ["🔥 STRONG BUY", "🟢 BUY SETUP"])]
                        import builtins as _bi
                        def _sf(_v, _default=0.0):
                            """Safe float conversion — uses builtins.float
                            explicitly (never shadowable) and handles
                            None/NaN/pandas-na gracefully instead of raising."""
                            try:
                                if _v is None:
                                    return _default
                                if isinstance(_v, str) and not _v.strip():
                                    return _default
                                if pd.isna(_v):
                                    return _default
                            except Exception:
                                pass
                            try:
                                return _bi.float(_v)
                            except Exception:
                                return _default
                        for _, _rw in _top_lg.iterrows():
                            db("INSERT INTO signal_log(user_id,log_date,scanner,"
                               "stock,score,entry,stop_loss,target,rr) "
                               "VALUES(?,?,?,?,?,?,?,?,?)",
                               (UID, _today_lg, "v2", str(_rw["Stock"]),
                                _sf(_rw["Score"]), _sf(_rw["Entry"]),
                                _sf(_rw["SL"]), _sf(_rw["Target"]),
                                _sf(_rw["RR"])))
                        st.toast(f"📸 Logged {len(_top_lg)} signals for {_today_lg}")
                    except Exception as _lge:
                        import traceback as _tb
                        st.error(f"Logging failed: {_lge}")
                        with st.expander("🔧 Full error detail (for debugging)"):
                            st.code(_tb.format_exc(), language=None)
            with lg2:
                st.caption("Log daily → within weeks you'll have real hit-rate data "
                           "per signal tier, so tuning becomes evidence-based.")

            with st.expander("📈 Logged Signal Performance", expanded=False):
                try:
                    _logs = db("SELECT log_date,stock,score,entry,stop_loss,target,rr "
                               "FROM signal_log WHERE user_id=? AND scanner=? "
                               "ORDER BY log_date DESC", (UID, "v2"), fetch=True)
                except Exception:
                    _logs = []
                if not _logs:
                    st.info("No logged signals yet — click 📸 after a scan.")
                else:
                    _lsyms = tuple(sorted({_l[1] for _l in _logs}))
                    _lp = _cached_prices(_lsyms)
                    _perf = []
                    for _ld, _st2, _sc2, _en, _slp, _tg, _rr2 in _logs:
                        _now_p = _lp.get(_st2)
                        _ret = (round((_now_p / _en - 1) * 100, 1)
                                if (_now_p and _en) else None)
                        _stat = "…"
                        if _now_p and _tg and _now_p >= _tg:   _stat = "🎯 Target"
                        elif _now_p and _slp and _now_p <= _slp: _stat = "🛑 Stopped"
                        _days = (datetime.now() -
                                 datetime.strptime(_ld, "%Y-%m-%d")).days
                        _perf.append({"Date": _ld, "Stock": _st2, "Days": _days,
                                      "Entry": _en, "Now": _now_p,
                                      "Ret %": _ret, "Status": _stat})
                    _pdf2 = pd.DataFrame(_perf)
                    st.dataframe(_pdf2, hide_index=True, use_container_width=True)
                    _rets = _pdf2["Ret %"].dropna()
                    if len(_rets) >= 3:
                        _wins2 = int((_rets > 0).sum())
                        st.markdown(
                            f'<div style="font-size:.85rem;font-weight:700">'
                            f'Hit rate: {_wins2}/{len(_rets)} '
                            f'({_wins2/len(_rets)*100:.0f}%) · '
                            f'Avg return: {_rets.mean():+.1f}%</div>',
                            unsafe_allow_html=True)
            st.caption("⚠️ Signals are algorithmic research candidates, not advice. "
                       "Bear regimes suppress buy tiers by design — that's the "
                       "system protecting you, not a bug.")

# ── Metrics ──────────────────────────────────────────────────────────────────
elif _page == 'metrics':
    a = calc_analytics(df)
    if not a or a.get("closed_trades", 0) == 0:
        st.info("Metrics require historical closed trades.")
    else:
        st.markdown('<div class="sec">Strategy Performance Metrics</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="cards">'
            + card("Win Rate",      f'{a["win_rate"]}%',
                   f'{a["wins"]}W / {a["losses"]}L',
                   "green" if a["win_rate"] >= 50 else "red")
            + card("Profit Factor", str(a["profit_factor"]), "Gross P / Gross L")
            + card("Expectancy",    f'₹{a["expectancy"]}')
            + card("Avg Win",       f'₹{a["avg_win"]:,.0f}')
            + card("Avg Loss",      f'₹{a["avg_loss"]:,.0f}', "", "red")
            + card("Max Drawdown",  f'₹{a["max_drawdown"]:,.0f}', "", "red")
            + card("Avg Hold",      f'{a["avg_hold_days"]}d')
            + card("Sharpe",        str(a["sharpe"]))
            + '</div>',
            unsafe_allow_html=True)

# ── Watchlist ────────────────────────────────────────────────────────────────
elif _page == 'watchlist':
    st.markdown('<div class="sec">👁 Target Watchlist</div>', unsafe_allow_html=True)

    def drop_watchlist_cb(w_id, s_name):
        delete_watchlist_item(w_id, UID)
        st.toast(f"🗑️ Dropped {s_name}")

    with st.form(key="add_stock_form", clear_on_submit=True):
        col_inp, col_btn = st.columns([4, 1])
        with col_inp:
            new_stock = st.text_input(
                "Stock Ticker", placeholder="e.g., SBIN, TATAMOTORS",
                label_visibility="collapsed").upper().strip()
        with col_btn:
            if st.form_submit_button("➕ Add", width="stretch") and new_stock:
                add_watchlist(UID, new_stock)
                st.toast(f"🚀 {new_stock} added!")
                st.rerun()

    wdf = get_watchlist(UID)
    if not wdf.empty:
        st.markdown('<div class="sec" style="margin-top:1rem">Live Monitored Assets</div>',
                    unsafe_allow_html=True)
        wl_symbols = wdf["stock"].tolist()
        with st.spinner("Fetching live metrics..."):
            wl_data = _bulk_fetch_history(wl_symbols, period="3mo")

        cols = st.columns(3)
        for i, row in wdf.iterrows():
            stock = row["stock"]
            wid   = int(row["id"])
            col   = cols[i % 3]
            with col:
                df_hist = wl_data.get(stock)
                ind = compute_indicators(stock, period="3mo", prefetched_df=df_hist)
                if ind:
                    cmp_v  = ind.get("cmp", "—")
                    rsi_v  = ind.get("rsi", "—")
                    trend  = ind.get("trend", "—")
                    sup    = ind.get("support", "—")
                    res    = ind.get("resistance", "—")
                    ema9   = ind.get("ema9",  ind.get("ema20", "—"))
                    ema21  = ind.get("ema21", ind.get("ema50", "—"))
                    brd = (theme_t["green"]  if "Uptrend"   in str(trend)
                           else theme_t["red"] if "Downtrend" in str(trend)
                           else theme_t["yellow"])
                    st.markdown(f"""
<div style="background:var(--card);border-top:4px solid {brd};border-radius:8px;
     padding:1rem;box-shadow:0 4px 6px rgba(0,0,0,.05);margin-bottom:.5rem">
  <div style="font-size:1.1rem;font-weight:800;color:var(--text)">{stock}</div>
  <div style="font-size:.75rem;color:var(--muted);margin-bottom:.5rem;
       text-transform:uppercase">{get_sector(stock)}</div>
  <div style="font-size:.8rem;line-height:1.6;color:var(--text)">
    <b>CMP:</b> ₹{cmp_v}<br>
    <b>RSI:</b> {rsi_v} | <b>Trend:</b> {trend}<br>
    <b>EMA9:</b> ₹{ema9} | <b>EMA21:</b> ₹{ema21}<br>
    <b>Sup:</b> ₹{sup} | <b>Res:</b> ₹{res}
  </div>
</div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
<div style="background:var(--card);border-top:4px solid var(--muted);
     border-radius:8px;padding:1rem;margin-bottom:.5rem">
  <div style="font-size:1.1rem;font-weight:800;color:var(--text)">{stock}</div>
  <div style="font-size:.85rem;color:var(--red);margin-bottom:.5rem">
    Data Unavailable — check symbol</div>
</div>""", unsafe_allow_html=True)

                st.button("🗑️ Drop", key=f"wl_del_{wid}",
                          on_click=drop_watchlist_cb,
                          args=(wid, stock), width="stretch")
    else:
        st.info("Watchlist empty. Add a ticker above.")

# ── Export ───────────────────────────────────────────────────────────────────
elif _page == 'export':
    if df.empty:
        st.info("No data available for export.")
    else:
        st.markdown('<div class="sec">Raw Database Export</div>',
                    unsafe_allow_html=True)
        st.dataframe(df)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV", csv,
            file_name=f"swing_portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv")

# ── Signal Scores ────────────────────────────────────────────────────────────
elif _page == 'scores':
    st.markdown('<div class="sec">🎯 signals.py — Component Scorecard</div>',
                unsafe_allow_html=True)
    render_score_dashboard()
    st.markdown("""
<div style="margin-top:1rem;padding:1rem;background:rgba(16,185,129,.08);
     border:1px solid rgba(16,185,129,.3);border-radius:8px;font-size:.85rem;
     color:var(--muted);line-height:1.8">
<b style="color:var(--text)">✅ Engine capabilities:</b><br>
1. <b>Core indicators</b> — Wilder RSI/ATR, single-pass MACD, numpy Supertrend, clamped Bollinger, 20-day VWAP, swing-peak Fibonacci<br>
2. <b>Risk engine</b> — unified <code>_calc_risk_params</code> across signals, picks, and scanner (zero phantom RR)<br>
3. <b>Trap scanner</b> — 5-factor bull/bear trap confluence across the full universe<br>
4. <b>Smart Money (SMC)</b> — FVG, order blocks, liquidity pools, premium/discount, displacement<br>
5. <b>VCP</b> — Minervini volatility-contraction base detection with pivot-ready flagging<br>
6. <b>Relative Strength</b> — IBD-style RS ratio + 1-99 percentile leadership rating vs Nifty<br>
7. <b>Unified in scanner</b> — VCP, Trap, and RS now surface as columns + filters in the Universe Scanner
</div>""", unsafe_allow_html=True)

# ── Trap Scanner ─────────────────────────────────────────────────────────────
elif _page == 'traps':
    if not _TRAP_SCANNER_AVAILABLE:
        st.warning("🪤 Trap Scanner requires the updated **signals.py** (v12+). "
                   "Deploy the new signals.py from the project outputs to enable this tab.",
                   icon="⚠️")
    st.markdown('<div class="sec">🪤 Bull & Bear Trap Scanner — Full Nifty 500</div>',
                unsafe_allow_html=True)

    # ── Summary banner ─────────────────────────────────────────────────────────
    trap_data = st.session_state.trap_scan_cache
    if trap_data:
        _stale_banner(trap_data.get("timestamp"), "🪤 Scan Traps")
        bull_n = trap_data.get("bull_count", 0)
        bear_n = trap_data.get("bear_count", 0)
        scanned = trap_data.get("scanned", 0)
        liquid  = trap_data.get("liquid", 0)
        ts      = trap_data.get("timestamp", "—")
        st.markdown(
            f'<div style="display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1.5rem">'
            f'<div style="background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);'
            f'border-radius:10px;padding:.7rem 1.2rem;font-weight:800;font-size:.9rem">'
            f'🔴 Bull Traps: <span style="color:var(--red)">{bull_n}</span></div>'
            f'<div style="background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.3);'
            f'border-radius:10px;padding:.7rem 1.2rem;font-weight:800;font-size:.9rem">'
            f'🟢 Bear Traps: <span style="color:var(--green)">{bear_n}</span></div>'
            f'<div style="background:var(--card);border:1px solid var(--border);'
            f'border-radius:10px;padding:.7rem 1.2rem;font-size:.8rem;color:var(--muted);font-weight:600">'
            f'🔍 Scanned: {scanned} | Liquid: {liquid} | Updated: {ts}</div>'
            f'</div>',
            unsafe_allow_html=True)

    # ── Controls ────────────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])
    with ctrl1:
        st.caption("⚡ Sweeps all Nifty 500 liquid stocks for false breakout / breakdown patterns.")
    with ctrl2:
        min_conf = st.slider("Min Confidence %", 50, 90, 60, 5, label_visibility="collapsed")
    with ctrl3:
        run_trap_scan = st.button("🪤 Run Trap Scan", width="stretch")

    if run_trap_scan:
        total_sym = len(SECTOR_MAP)
        with st.spinner(f"🔍 Scanning {total_sym} stocks for trap patterns…"):
            st.session_state.trap_scan_cache = scan_for_traps(min_confidence=min_conf)
            trap_data = st.session_state.trap_scan_cache
            st.toast(
                f"✅ Found {trap_data['bull_count']} bull traps, "
                f"{trap_data['bear_count']} bear traps across {trap_data['liquid']} liquid stocks",
                icon="🪤")

    if not trap_data:
        st.info("💡 Click **🪤 Run Trap Scan** to sweep the full Nifty 500 for active trap patterns.")
    else:
        bull_traps = trap_data.get("bull_traps", [])
        bear_traps = trap_data.get("bear_traps", [])

        # ── Filter by confidence slider ─────────────────────────────────────────
        bull_traps = [x for x in bull_traps if x["confidence"] >= min_conf]
        bear_traps = [x for x in bear_traps if x["confidence"] >= min_conf]

        col_bull, col_bear = st.columns(2)

        # ── BULL TRAPS ──────────────────────────────────────────────────────────
        with col_bull:
            st.markdown(
                f'<div style="font-size:.85rem;font-weight:800;color:var(--red);'
                f'text-transform:uppercase;letter-spacing:.1em;margin-bottom:.8rem;'
                f'padding:.5rem .8rem;background:rgba(239,68,68,.08);'
                f'border-left:4px solid var(--red);border-radius:0 8px 8px 0">'
                f'🔴 Bull Traps — Exit / Avoid ({len(bull_traps)})</div>',
                unsafe_allow_html=True)

            if not bull_traps:
                st.success("✅ No bull traps found at this confidence level.")
            else:
                for bt in bull_traps:
                  try:
                    conf = bt["confidence"]
                    conf_clr = "#ef4444" if conf >= 80 else "#f59e0b"
                    st.markdown(f"""
<div style="background:var(--card);border:1px solid rgba(239,68,68,.25);
     border-left:4px solid var(--red);border-radius:10px;
     padding:1rem 1.2rem;margin-bottom:.8rem;
     box-shadow:0 4px 12px -4px rgba(239,68,68,.15)">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.4rem">
    <span style="font-weight:800;font-size:.95rem">{bt['stock']}</span>
    <span style="background:rgba(239,68,68,.12);color:{conf_clr};
          padding:.2rem .6rem;border-radius:6px;font-size:.75rem;font-weight:800">
      {conf}% CONF
    </span>
  </div>
  <div style="font-size:.75rem;color:var(--muted);margin-bottom:.5rem">
    {bt['sector']} · CMP ₹{bt['cmp']} · RSI {bt['rsi'] if bt['rsi'] else '—'}
  </div>
  <div style="font-size:.8rem;color:var(--red);font-weight:600;margin-bottom:.5rem">
    ⚠️ {bt['detail']}
  </div>
  <div style="height:4px;background:var(--input);border-radius:2px;margin-bottom:.6rem">
    <div style="height:4px;border-radius:2px;background:var(--red);width:{min(conf,100)}%"></div>
  </div>
  <div style="font-size:.78rem;color:var(--muted);display:grid;grid-template-columns:1fr 1fr;gap:.2rem">
    <span>📊 Trend: {bt['trend']}</span>
    <span>📦 Vol: {bt['vol_ratio']:.1f}x avg</span>
    <span>🛡 Support: ₹{bt['support']}</span>
    <span>🚧 Resist: ₹{bt['resistance']}</span>
    <span>🔁 Re-entry SL: ₹{bt['re_entry_sl']}</span>
    <span>ST: {'🟢 Bull' if bt.get('supertrend_bullish') else '🔴 Bear'}</span>
  </div>
  {('<div style="font-size:.72rem;color:var(--muted);margin-top:.4rem">📐 ' + bt['patterns'] + '</div>') if bt.get('patterns') else ''}
</div>""", unsafe_allow_html=True)
                  except Exception:
                    st.markdown(
                        f'<div style="background:var(--card);border:1px solid var(--border);'
                        f'border-radius:8px;padding:.7rem 1rem;margin-bottom:.6rem;font-size:.85rem">'
                        f'<b>{bt.get("stock","?")}</b> — bull trap '
                        f'{bt.get("confidence","?")}% (detail unavailable)</div>',
                        unsafe_allow_html=True)

        # ── BEAR TRAPS ──────────────────────────────────────────────────────────
        with col_bear:
            st.markdown(
                f'<div style="font-size:.85rem;font-weight:800;color:var(--green);'
                f'text-transform:uppercase;letter-spacing:.1em;margin-bottom:.8rem;'
                f'padding:.5rem .8rem;background:rgba(16,185,129,.08);'
                f'border-left:4px solid var(--green);border-radius:0 8px 8px 0">'
                f'🟢 Bear Traps — Buy Opportunity ({len(bear_traps)})</div>',
                unsafe_allow_html=True)

            if not bear_traps:
                st.info("No bear traps found at this confidence level.")
            else:
                for brt in bear_traps:
                  try:
                    conf = brt["confidence"]
                    conf_clr = "#10b981" if conf >= 80 else "#f59e0b"
                    rr = brt.get("risk_reward")
                    rr_str = f"R:R {rr}" if rr else "—"
                    st.markdown(f"""
<div style="background:var(--card);border:1px solid rgba(16,185,129,.25);
     border-left:4px solid var(--green);border-radius:10px;
     padding:1rem 1.2rem;margin-bottom:.8rem;
     box-shadow:0 4px 12px -4px rgba(16,185,129,.15)">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.4rem">
    <span style="font-weight:800;font-size:.95rem">{brt['stock']}</span>
    <span style="background:rgba(16,185,129,.12);color:{conf_clr};
          padding:.2rem .6rem;border-radius:6px;font-size:.75rem;font-weight:800">
      {conf}% CONF
    </span>
  </div>
  <div style="font-size:.75rem;color:var(--muted);margin-bottom:.5rem">
    {brt['sector']} · CMP ₹{brt['cmp']} · RSI {brt['rsi'] if brt['rsi'] else '—'}
  </div>
  <div style="font-size:.8rem;color:var(--green);font-weight:600;margin-bottom:.5rem">
    🪤 {brt['detail']}
  </div>
  <div style="height:4px;background:var(--input);border-radius:2px;margin-bottom:.6rem">
    <div style="height:4px;border-radius:2px;background:var(--green);width:{min(conf,100)}%"></div>
  </div>
  <div style="background:rgba(16,185,129,.06);border-radius:6px;
       padding:.6rem .8rem;margin-bottom:.5rem;
       display:grid;grid-template-columns:1fr 1fr 1fr;gap:.3rem;font-size:.8rem;font-weight:700">
    <span>🎯 Entry<br><b>₹{brt['entry']}</b></span>
    <span>🚀 Target<br><b style="color:var(--green)">₹{brt['target']}</b></span>
    <span>🛑 SL<br><b style="color:var(--red)">₹{brt['stop_loss']}</b></span>
  </div>
  <div style="font-size:.78rem;color:var(--muted);display:grid;grid-template-columns:1fr 1fr;gap:.2rem">
    <span>📊 {rr_str}</span>
    <span>📦 Vol: {brt['vol_ratio']:.1f}x avg</span>
    <span>🛡 Support: ₹{brt['support']}</span>
    <span>🚧 Resist: ₹{brt['resistance']}</span>
    <span>📈 Trend: {brt['trend']}</span>
    <span>ST: {'🟢 Bull' if brt.get('supertrend_bullish') else '🔴 Bear'}</span>
  </div>
  {('<div style="font-size:.72rem;color:var(--muted);margin-top:.4rem">📐 ' + brt['patterns'] + '</div>') if brt.get('patterns') else ''}
</div>""", unsafe_allow_html=True)
                  except Exception:
                    st.markdown(
                        f'<div style="background:var(--card);border:1px solid var(--border);'
                        f'border-radius:8px;padding:.7rem 1rem;margin-bottom:.6rem;font-size:.85rem">'
                        f'<b>{brt.get("stock","?")}</b> — bear trap '
                        f'{brt.get("confidence","?")}% (detail unavailable)</div>',
                        unsafe_allow_html=True)

        # ── Export trap results ─────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        if bull_traps or bear_traps:
            bull_df = pd.DataFrame(bull_traps)[
                ["stock","sector","cmp","rsi","confidence","detail","support","resistance","trend"]
            ] if bull_traps else pd.DataFrame()
            bear_df = pd.DataFrame(bear_traps)[
                ["stock","sector","cmp","rsi","confidence","detail","entry","target","stop_loss","risk_reward","trend"]
            ] if bear_traps else pd.DataFrame()

            exp1, exp2 = st.columns(2)
            with exp1:
                if not bull_df.empty:
                    st.download_button(
                        "⬇️ Export Bull Traps CSV",
                        bull_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"bull_traps_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv", use_container_width=True)
            with exp2:
                if not bear_df.empty:
                    st.download_button(
                        "⬇️ Export Bear Traps CSV",
                        bear_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"bear_traps_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv", use_container_width=True)

# ── Corporate Actions ────────────────────────────────────────────────────────
elif _page == 'corp_actions':
    if not _CORP_ACTIONS_AVAILABLE:
        st.warning("📅 Corporate Actions requires the updated **signals.py** (v12+). "
                   "Deploy the new signals.py from the project outputs to enable this tab.",
                   icon="⚠️")
    st.markdown('<div class="sec">📅 Corporate Actions — Full Nifty 500</div>',
                unsafe_allow_html=True)
    st.caption("Dividends · Stock Splits · Bonus Issues — sourced from NSE via yfinance. 6-hour cache.")

    # ── Portfolio holdings quick-view ──────────────────────────────────────────
    open_syms = raw[raw["status"]=="Open"]["stock"].unique().tolist() if not raw.empty else []
    if open_syms:
        st.markdown('<div class="sec" style="margin-top:1rem">📌 Your Holdings</div>',
                    unsafe_allow_html=True)
        with st.spinner("Fetching corporate actions for your holdings..."):
            port_actions = fetch_bulk_corporate_actions(open_syms, max_workers=5)

        p_rows = ""
        for sym in open_syms:
            data = port_actions.get(sym, {})
            div_str  = (f"₹{data['last_dividend']} on {data['last_div_date']}"
                        if data.get("last_dividend") else "—")
            exd_str  = (f'<b style="color:var(--yellow)">{data["upcoming_exdate"]}</b>'
                        if data.get("upcoming_exdate") else "—")
            spl_str  = (f"{data['splits'][-1]['ratio']}x on {data['splits'][-1]['date']}"
                        if data.get("splits") else "—")
            split_badge = ('<span style="background:rgba(59,130,246,.15);color:var(--blue);'
                           'padding:.1rem .5rem;border-radius:4px;font-size:.7rem;'
                           'font-weight:800">SPLIT/BONUS 1Y</span>'
                           if data.get("has_split_1y") else "")
            p_rows += (
                f"<tr>"
                f"<td style='text-align:left;font-weight:800'>{sym} {split_badge}</td>"
                f"<td style='text-align:left;color:var(--muted);font-size:.8rem'>"
                f"{get_sector(sym)}</td>"
                f"<td>{div_str}</td>"
                f"<td>{exd_str}</td>"
                f"<td style='font-size:.78rem;color:var(--muted)'>{spl_str}</td>"
                f"</tr>"
            )
        if p_rows:
            st.markdown(
                f'<div class="tbl-wrap"><table class="t">'
                f'<thead><tr>'
                f'<th class="l">Stock</th><th class="l">Sector</th>'
                f'<th>Last Dividend</th><th>Ex-Date (upcoming)</th>'
                f'<th>Last Split/Bonus</th>'
                f'</tr></thead><tbody>{p_rows}</tbody></table></div>',
                unsafe_allow_html=True)

    st.markdown("<hr style='border-color:var(--border);margin:1.5rem 0'>",
                unsafe_allow_html=True)

    # ── Full universe scan controls ────────────────────────────────────────────
    ca1, ca2 = st.columns([3, 1])
    with ca1:
        st.markdown(
            '<div style="font-size:.85rem;font-weight:700;color:var(--text)">'
            '🔍 Sweep full Nifty 500 for upcoming ex-dates, recent dividends '
            'and bonus/split events</div>',
            unsafe_allow_html=True)
    with ca2:
        run_ca_scan = st.button("📅 Scan Corporate Actions", width="stretch")

    if run_ca_scan:
        total_sym = len(SECTOR_MAP)
        with st.spinner(f"Fetching corporate actions for {total_sym} stocks… (may take 60–90s)"):
            st.session_state.corp_actions_cache = scan_corporate_actions_universe()
            ca = st.session_state.corp_actions_cache
            st.toast(
                f"✅ {len(ca['with_upcoming_exdate'])} upcoming ex-dates · "
                f"{len(ca['recent_dividends'])} recent dividends · "
                f"{len(ca['recent_splits'])} splits/bonus",
                icon="📅")

    ca_data = st.session_state.corp_actions_cache
    if ca_data is None:
        st.info("Click **📅 Scan Corporate Actions** to fetch the full Nifty 500 action calendar.")
    else:
        ts = ca_data.get("timestamp","—"); scanned = ca_data.get("scanned",0)
        st.markdown(
            f'<div style="font-size:.75rem;color:var(--muted);margin-bottom:1rem">'
            f'Last scanned {scanned} stocks at {ts}</div>',
            unsafe_allow_html=True)

        ca_t1, ca_t2, ca_t3 = st.tabs([
            f"📆 Upcoming Ex-Dates ({len(ca_data['with_upcoming_exdate'])})",
            f"💰 Recent Dividends ({len(ca_data['recent_dividends'])})",
            f"🔀 Splits & Bonus ({len(ca_data['recent_splits'])})",
        ])

        # ── Upcoming Ex-Dates ──────────────────────────────────────────────────
        with ca_t1:
            exd_list = ca_data["with_upcoming_exdate"]
            if not exd_list:
                st.info("No upcoming ex-dividend dates found.")
            else:
                rows = ""
                for item in exd_list:
                    days_away = (pd.Timestamp(item["ex_date"]) -
                                 pd.Timestamp.now()).days
                    urgency = (
                        f'<span style="color:var(--red);font-weight:800">'
                        f'⚡ {days_away}d away</span>'
                        if days_away <= 7 else
                        f'<span style="color:var(--yellow);font-weight:700">'
                        f'{days_away}d</span>'
                        if days_away <= 30 else
                        f'<span style="color:var(--muted)">{days_away}d</span>'
                    )
                    div_amt = (f"₹{item['last_dividend']}"
                               if item.get("last_dividend") else "—")
                    rows += (
                        f"<tr>"
                        f"<td style='text-align:left;font-weight:800'>{item['stock']}</td>"
                        f"<td style='text-align:left;color:var(--muted);font-size:.8rem'>"
                        f"{item['sector']}</td>"
                        f"<td><b>{item['ex_date']}</b></td>"
                        f"<td>{urgency}</td>"
                        f"<td>{div_amt}</td>"
                        f"</tr>"
                    )
                st.markdown(
                    f'<div class="tbl-wrap"><table class="t"><thead><tr>'
                    f'<th class="l">Stock</th><th class="l">Sector</th>'
                    f'<th>Ex-Date</th><th>Days Away</th><th>Last Div Amt</th>'
                    f'</tr></thead><tbody>{rows}</tbody></table></div>',
                    unsafe_allow_html=True)
                # Export
                ex_df = pd.DataFrame(exd_list)
                st.download_button(
                    "⬇️ Export Ex-Dates CSV",
                    ex_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"upcoming_exdates_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv")

        # ── Recent Dividends ───────────────────────────────────────────────────
        with ca_t2:
            div_list = ca_data["recent_dividends"]
            if not div_list:
                st.info("No recent dividends found in the last 12 months.")
            else:
                rows = ""
                for item in sorted(div_list, key=lambda x: x["amount"], reverse=True):
                    rows += (
                        f"<tr>"
                        f"<td style='text-align:left;font-weight:800'>{item['stock']}</td>"
                        f"<td style='text-align:left;color:var(--muted);font-size:.8rem'>"
                        f"{item['sector']}</td>"
                        f"<td><b style='color:var(--green)'>₹{item['amount']}</b></td>"
                        f"<td>{item['ex_date']}</td>"
                        f"</tr>"
                    )
                st.markdown(
                    f'<div class="tbl-wrap"><table class="t"><thead><tr>'
                    f'<th class="l">Stock</th><th class="l">Sector</th>'
                    f'<th>Dividend ₹</th><th>Ex-Date</th>'
                    f'</tr></thead><tbody>{rows}</tbody></table></div>',
                    unsafe_allow_html=True)
                div_df = pd.DataFrame(div_list)
                st.download_button(
                    "⬇️ Export Dividends CSV",
                    div_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"recent_dividends_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv")

        # ── Splits & Bonus ─────────────────────────────────────────────────────
        with ca_t3:
            split_list = ca_data["recent_splits"]
            if not split_list:
                st.info("No stock splits or bonus issues found in the last 12 months.")
            else:
                rows = ""
                for item in split_list:
                    type_badge = (
                        f'<span style="background:rgba(59,130,246,.15);color:var(--blue);'
                        f'padding:.2rem .6rem;border-radius:5px;font-size:.72rem;'
                        f'font-weight:800">{item["type"]}</span>'
                    )
                    rows += (
                        f"<tr>"
                        f"<td style='text-align:left;font-weight:800'>{item['stock']}</td>"
                        f"<td style='text-align:left;color:var(--muted);font-size:.8rem'>"
                        f"{item['sector']}</td>"
                        f"<td>{type_badge}</td>"
                        f"<td><b>{item['ratio']}:1</b></td>"
                        f"<td>{item['date']}</td>"
                        f"</tr>"
                    )
                st.markdown(
                    f'<div class="tbl-wrap"><table class="t"><thead><tr>'
                    f'<th class="l">Stock</th><th class="l">Sector</th>'
                    f'<th>Type</th><th>Ratio</th><th>Date</th>'
                    f'</tr></thead><tbody>{rows}</tbody></table></div>',
                    unsafe_allow_html=True)
                sp_df = pd.DataFrame(split_list)
                st.download_button(
                    "⬇️ Export Splits/Bonus CSV",
                    sp_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"splits_bonus_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv")

# ── Smart Money Concepts (SMC / ICT) ──────────────────────────────────────────
elif _page == 'smc':
    st.markdown('<div class="sec">🏦 Smart Money Concepts — FVG · Order Blocks · Liquidity</div>',
                unsafe_allow_html=True)
    st.caption("Institutional footprint analysis: Fair Value Gaps, Order Blocks, "
               "Liquidity Pools, Premium/Discount zones, and Displacement. "
               "Optimised for NSE daily charts with circuit-filter awareness.")

    # ── Stock selector: portfolio holdings + custom symbol ─────────────────────
    open_syms = (raw[raw["status"]=="Open"]["stock"].unique().tolist()
                 if not raw.empty else [])
    sc1, sc2 = st.columns([2, 1])
    with sc1:
        symbol_options = open_syms + ["— Enter custom symbol —"]
        sel_sym = st.selectbox("Select stock for SMC analysis", symbol_options,
                               label_visibility="collapsed")
    with sc2:
        custom_sym = st.text_input("Custom", placeholder="e.g. RELIANCE",
                                   label_visibility="collapsed")

    target_sym = (custom_sym.strip().upper() if custom_sym.strip()
                  else (sel_sym if sel_sym != "— Enter custom symbol —" else None))

    if not target_sym:
        st.info("💡 Select a holding or enter any NSE symbol to see its Smart Money structure.")
    else:
        with st.spinner(f"Analysing {target_sym} institutional structure…"):
            try:
                ind = compute_indicators(target_sym, period="6mo")
            except Exception as e:
                ind = None
                st.error(f"Could not analyse {target_sym}: {e}")

        if ind:
            cmp = ind.get("cmp", 0)
            score = ind.get("smc_score", 0)
            label = ind.get("smc_label", "Neutral SMC")
            zone  = ind.get("smc_zone", "Unknown")
            bias  = ind.get("smc_bias", "Neutral")
            action = ind.get("smc_action", "WAIT")
            entry  = ind.get("smc_entry")
            target = ind.get("smc_target")
            sl     = ind.get("smc_sl")
            rr     = ind.get("smc_rr")
            quality = ind.get("smc_setup_quality")
            reason = ind.get("smc_setup_reason", "")

            # ── ACTIONABLE TRADE SETUP CARD (the headline) ─────────────────────
            if action in ("BUY", "SELL") and entry:
                act_clr = theme_t["green"] if action == "BUY" else theme_t["red"]
                act_bg  = ("rgba(16,185,129,.08)" if action == "BUY"
                           else "rgba(239,68,68,.08)")
                q_clr = ("#fbbf24" if quality == "A+" else
                         theme_t["accent"] if quality == "A" else theme_t["muted"])
                st.markdown(
                    f'<div style="background:{act_bg};border:2px solid {act_clr};'
                    f'border-radius:14px;padding:1.4rem;margin:1rem 0">'
                    f'<div style="display:flex;justify-content:space-between;'
                    f'align-items:center;margin-bottom:1rem">'
                    f'<div style="font-size:1.8rem;font-weight:800;color:{act_clr}">'
                    f'{"🟢" if action=="BUY" else "🔴"} {action} {target_sym}</div>'
                    f'<div style="background:{q_clr};color:#000;padding:.3rem .9rem;'
                    f'border-radius:8px;font-size:.95rem;font-weight:800">'
                    f'{quality} Setup</div></div>'
                    f'<div style="display:grid;grid-template-columns:repeat(4,1fr);'
                    f'gap:.8rem;margin-bottom:1rem">'
                    f'<div style="background:var(--card);border-radius:10px;padding:.9rem;'
                    f'text-align:center"><div style="font-size:.7rem;color:var(--muted);'
                    f'font-weight:700;text-transform:uppercase">Entry</div>'
                    f'<div style="font-size:1.3rem;font-weight:800;color:var(--text)">'
                    f'₹{entry}</div></div>'
                    f'<div style="background:var(--card);border-radius:10px;padding:.9rem;'
                    f'text-align:center"><div style="font-size:.7rem;color:var(--muted);'
                    f'font-weight:700;text-transform:uppercase">Target</div>'
                    f'<div style="font-size:1.3rem;font-weight:800;color:{theme_t["green"]}">'
                    f'₹{target}</div></div>'
                    f'<div style="background:var(--card);border-radius:10px;padding:.9rem;'
                    f'text-align:center"><div style="font-size:.7rem;color:var(--muted);'
                    f'font-weight:700;text-transform:uppercase">Stop Loss</div>'
                    f'<div style="font-size:1.3rem;font-weight:800;color:{theme_t["red"]}">'
                    f'₹{sl}</div></div>'
                    f'<div style="background:var(--card);border-radius:10px;padding:.9rem;'
                    f'text-align:center"><div style="font-size:.7rem;color:var(--muted);'
                    f'font-weight:700;text-transform:uppercase">Risk:Reward</div>'
                    f'<div style="font-size:1.3rem;font-weight:800;color:var(--accent)">'
                    f'1:{rr}</div></div>'
                    f'</div>'
                    f'<div style="font-size:.82rem;color:var(--muted);line-height:1.6">'
                    f'📋 {reason}</div>'
                    f'</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div style="background:var(--card);border:2px solid var(--border);'
                    f'border-radius:14px;padding:1.4rem;margin:1rem 0;text-align:center">'
                    f'<div style="font-size:1.5rem;font-weight:800;color:var(--muted)">'
                    f'⏸ WAIT — {target_sym}</div>'
                    f'<div style="font-size:.85rem;color:var(--muted);margin-top:.5rem">'
                    f'{reason}</div></div>',
                    unsafe_allow_html=True)

            # ── Context cards (now secondary, below the action) ────────────────
            score_clr = (theme_t["green"] if score >= 35 else
                         theme_t["red"] if score <= -35 else theme_t["muted"])
            zone_clr  = (theme_t["red"] if zone == "Premium" else
                         theme_t["green"] if zone == "Discount" else theme_t["muted"])
            st.markdown(
                f'<div style="display:flex;gap:1rem;flex-wrap:wrap;margin:1rem 0">'
                f'<div style="flex:1;min-width:180px;background:var(--card);'
                f'border:1px solid {score_clr};border-radius:12px;padding:1.2rem">'
                f'<div style="font-size:.7rem;color:var(--muted);font-weight:700;'
                f'text-transform:uppercase;letter-spacing:.08em">SMC Bias</div>'
                f'<div style="font-size:1.5rem;font-weight:800;color:{score_clr};'
                f'margin:.2rem 0">{label}</div>'
                f'<div style="font-size:.8rem;color:var(--muted)">Score: {score:+d} / 100</div>'
                f'</div>'
                f'<div style="flex:1;min-width:180px;background:var(--card);'
                f'border:1px solid {zone_clr};border-radius:12px;padding:1.2rem">'
                f'<div style="font-size:.7rem;color:var(--muted);font-weight:700;'
                f'text-transform:uppercase;letter-spacing:.08em">Premium/Discount</div>'
                f'<div style="font-size:1.5rem;font-weight:800;color:{zone_clr};'
                f'margin:.2rem 0">{zone}</div>'
                f'<div style="font-size:.8rem;color:var(--muted)">'
                f'{ind.get("smc_zone_pct","—")}% of range · {bias} bias</div>'
                f'</div>'
                f'<div style="flex:1;min-width:180px;background:var(--card);'
                f'border:1px solid var(--border);border-radius:12px;padding:1.2rem">'
                f'<div style="font-size:.7rem;color:var(--muted);font-weight:700;'
                f'text-transform:uppercase;letter-spacing:.08em">CMP</div>'
                f'<div style="font-size:1.5rem;font-weight:800;color:var(--text);'
                f'margin:.2rem 0">₹{cmp}</div>'
                f'<div style="font-size:.8rem;color:var(--muted)">'
                f'Range ₹{ind.get("smc_range_low","—")}–₹{ind.get("smc_range_high","—")}</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True)

            # ── Displacement banner ────────────────────────────────────────────
            disp = ind.get("smc_displacement")
            if disp:
                d_ago = ind.get("smc_displacement_bars_ago", "?")
                d_clr = theme_t["green"] if disp == "Bullish" else theme_t["red"]
                st.markdown(
                    f'<div style="background:rgba(0,0,0,.15);border-left:4px solid {d_clr};'
                    f'border-radius:0 8px 8px 0;padding:.7rem 1rem;margin-bottom:1rem;'
                    f'font-size:.85rem">⚡ <b style="color:{d_clr}">{disp} Displacement</b> '
                    f'detected {d_ago} bar(s) ago — institutional momentum present.</div>',
                    unsafe_allow_html=True)

            # ── Four detail panels ─────────────────────────────────────────────
            colA, colB = st.columns(2)

            # FVG panel
            with colA:
                st.markdown('<div style="font-size:.85rem;font-weight:800;color:var(--text);'
                            'margin-bottom:.5rem">📊 Fair Value Gaps</div>',
                            unsafe_allow_html=True)
                nbf = ind.get("smc_nearest_bull_fvg")
                nbef = ind.get("smc_nearest_bear_fvg")
                in_bull = ind.get("smc_in_bull_fvg")
                in_bear = ind.get("smc_in_bear_fvg")
                fvg_rows = ""
                if in_bull:
                    fvg_rows += ('<div style="color:var(--green);font-size:.82rem;'
                                 'margin-bottom:.3rem">📍 Price currently INSIDE a bullish FVG (support)</div>')
                if in_bear:
                    fvg_rows += ('<div style="color:var(--red);font-size:.82rem;'
                                 'margin-bottom:.3rem">📍 Price currently INSIDE a bearish FVG (resistance)</div>')
                if nbf:
                    fvg_rows += (f'<div style="font-size:.82rem;margin-bottom:.3rem">'
                                 f'🟢 Nearest bull FVG below: <b>₹{nbf["bottom"]}–₹{nbf["top"]}</b> '
                                 f'({nbf["size_atr"]} ATR)</div>')
                if nbef:
                    fvg_rows += (f'<div style="font-size:.82rem;margin-bottom:.3rem">'
                                 f'🔴 Nearest bear FVG above: <b>₹{nbef["bottom"]}–₹{nbef["top"]}</b> '
                                 f'({nbef["size_atr"]} ATR)</div>')
                fvg_rows += (f'<div style="font-size:.75rem;color:var(--muted);margin-top:.4rem">'
                             f'Unfilled: {ind.get("smc_bull_fvg_count",0)} bullish · '
                             f'{ind.get("smc_bear_fvg_count",0)} bearish</div>')
                if not (nbf or nbef or in_bull or in_bear):
                    fvg_rows = '<div style="font-size:.82rem;color:var(--muted)">No significant unfilled FVGs nearby.</div>'
                st.markdown(f'<div style="background:var(--card);border:1px solid var(--border);'
                            f'border-radius:10px;padding:1rem">{fvg_rows}</div>',
                            unsafe_allow_html=True)

            # Order Block panel
            with colB:
                st.markdown('<div style="font-size:.85rem;font-weight:800;color:var(--text);'
                            'margin-bottom:.5rem">🧱 Order Blocks</div>',
                            unsafe_allow_html=True)
                nbo = ind.get("smc_nearest_bull_ob")
                nbeo = ind.get("smc_nearest_bear_ob")
                ob_rows = ""
                if ind.get("smc_at_bull_ob"):
                    ob_rows += ('<div style="color:var(--green);font-size:.82rem;'
                                'margin-bottom:.3rem">📍 Price at a bullish order block (demand)</div>')
                if ind.get("smc_at_bear_ob"):
                    ob_rows += ('<div style="color:var(--red);font-size:.82rem;'
                                'margin-bottom:.3rem">📍 Price at a bearish order block (supply)</div>')
                if nbo:
                    ob_rows += (f'<div style="font-size:.82rem;margin-bottom:.3rem">'
                                f'🟢 Bull OB (demand): <b>₹{nbo["bottom"]}–₹{nbo["top"]}</b> '
                                f'({nbo["strength_atr"]} ATR move)</div>')
                if nbeo:
                    ob_rows += (f'<div style="font-size:.82rem;margin-bottom:.3rem">'
                                f'🔴 Bear OB (supply): <b>₹{nbeo["bottom"]}–₹{nbeo["top"]}</b> '
                                f'({nbeo["strength_atr"]} ATR move)</div>')
                if not (nbo or nbeo or ind.get("smc_at_bull_ob") or ind.get("smc_at_bear_ob")):
                    ob_rows = '<div style="font-size:.82rem;color:var(--muted)">No active order blocks nearby.</div>'
                st.markdown(f'<div style="background:var(--card);border:1px solid var(--border);'
                            f'border-radius:10px;padding:1rem">{ob_rows}</div>',
                            unsafe_allow_html=True)

            colC, colD = st.columns(2)

            # Liquidity panel
            with colC:
                st.markdown('<div style="font-size:.85rem;font-weight:800;color:var(--text);'
                            'margin:.8rem 0 .5rem">💧 Liquidity Pools</div>',
                            unsafe_allow_html=True)
                nbs = ind.get("smc_nearest_buyside")
                nss = ind.get("smc_nearest_sellside")
                liq_rows = ""
                if nbs:
                    liq_rows += (f'<div style="font-size:.82rem;margin-bottom:.3rem">'
                                 f'🔼 Buy-side liquidity above: <b>₹{nbs["level"]}</b> '
                                 f'({nbs["touches"]} equal highs — short stops)</div>')
                if nss:
                    liq_rows += (f'<div style="font-size:.82rem;margin-bottom:.3rem">'
                                 f'🔽 Sell-side liquidity below: <b>₹{nss["level"]}</b> '
                                 f'({nss["touches"]} equal lows — long stops)</div>')
                if not (nbs or nss):
                    liq_rows = '<div style="font-size:.82rem;color:var(--muted)">No clear liquidity clusters nearby.</div>'
                st.markdown(f'<div style="background:var(--card);border:1px solid var(--border);'
                            f'border-radius:10px;padding:1rem">{liq_rows}</div>',
                            unsafe_allow_html=True)

            # How to read panel
            with colD:
                st.markdown('<div style="font-size:.85rem;font-weight:800;color:var(--text);'
                            'margin:.8rem 0 .5rem">📖 How to Read This</div>',
                            unsafe_allow_html=True)
                st.markdown(
                    '<div style="background:var(--card);border:1px solid var(--border);'
                    'border-radius:10px;padding:1rem;font-size:.78rem;color:var(--muted);'
                    'line-height:1.7">'
                    '<b style="color:var(--text)">Confluence is key:</b> a bullish setup is '
                    'strongest when price is in <b>Discount</b>, sitting at a <b>bull Order Block</b> '
                    'or <b>FVG</b>, with recent <b>bullish Displacement</b>. '
                    'Liquidity pools show where price is likely drawn next (stop hunts).'
                    '</div>',
                    unsafe_allow_html=True)

            # ── Confluence with existing signals ───────────────────────────────
            st.markdown('<div style="font-size:.85rem;font-weight:800;color:var(--text);'
                        'margin:1.2rem 0 .5rem">🔗 Confluence with Technical Signals</div>',
                        unsafe_allow_html=True)
            conf_items = []
            if ind.get("bull_trap"):
                conf_items.append(("🪤 Bull Trap active", "bear"))
            if ind.get("bear_trap"):
                conf_items.append(("🪤 Bear Trap active", "bull"))
            if ind.get("supertrend_bullish"): conf_items.append(("Supertrend Bullish", "bull"))
            else: conf_items.append(("Supertrend Bearish", "bear"))
            if ind.get("rsi"):
                if ind["rsi"] >= 70: conf_items.append((f"RSI Overbought ({ind['rsi']})", "bear"))
                elif ind["rsi"] <= 30: conf_items.append((f"RSI Oversold ({ind['rsi']})", "bull"))
            if score >= 35: conf_items.append(("SMC Bullish Confluence", "bull"))
            elif score <= -35: conf_items.append(("SMC Bearish Confluence", "bear"))

            chips = ""
            for txt, side in conf_items:
                c = theme_t["green"] if side == "bull" else theme_t["red"]
                chips += (f'<span style="background:rgba(0,0,0,.12);border:1px solid {c};'
                          f'color:{c};border-radius:6px;padding:.3rem .7rem;font-size:.78rem;'
                          f'font-weight:700;margin:.2rem">{txt}</span> ')
            st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:.3rem">{chips}</div>',
                        unsafe_allow_html=True)

    # ── Universe-wide SMC setup scanner ────────────────────────────────────────
    st.markdown("<hr style='border-color:var(--border);margin:1.5rem 0'>",
                unsafe_allow_html=True)
    st.markdown('<div class="sec">🎯 Scan Universe for SMC Setups</div>',
                unsafe_allow_html=True)

    if not _SMC_SCANNER_AVAILABLE:
        st.warning("Universe SMC scan requires the updated signals.py (with "
                   "scan_for_smc_setups). Deploy the latest signals.py to enable.",
                   icon="⚠️")
    else:
        scs1, scs2, scs3 = st.columns([1.2, 1.2, 1])
        with scs1:
            min_q = st.selectbox("Min quality", ["B", "A", "A+"],
                                 label_visibility="collapsed")
        with scs2:
            act_f = st.selectbox("Action", ["All", "BUY", "SELL"],
                                 label_visibility="collapsed")
        with scs3:
            run_smc_scan = st.button("🎯 Scan Setups", width="stretch")

        if run_smc_scan:
            with st.spinner(f"Scanning {len(SECTOR_MAP)} stocks for SMC setups…"):
                st.session_state.smc_scan_cache = scan_for_smc_setups(
                    min_quality=min_q, action_filter=act_f)
                sc = st.session_state.smc_scan_cache
                st.toast(f"✅ {sc['buy_count']} BUY · {sc['sell_count']} SELL setups",
                         icon="🎯")

        sc = st.session_state.get("smc_scan_cache")
        if sc:
            st.markdown(
                f'<div style="font-size:.75rem;color:var(--muted);margin:.5rem 0">'
                f'Scanned {sc["scanned"]} · {sc["liquid"]} liquid · '
                f'{sc["buy_count"]} BUY · {sc["sell_count"]} SELL · {sc["timestamp"]}</div>',
                unsafe_allow_html=True)

            all_setups = sc["buy_setups"] + sc["sell_setups"]
            if all_setups:
                rows = []
                for s in all_setups:
                    rows.append({
                        "Stock": s["stock"], "Sector": s["sector"],
                        "Action": s["action"], "Grade": s["quality"],
                        "CMP": s["cmp"], "Entry": s["entry"],
                        "Target": s["target"], "SL": s["stop_loss"],
                        "RR": s["risk_reward"], "Zone": s["zone"],
                        "SMC": s["smc_score"],
                    })
                setup_df = pd.DataFrame(rows)
                # Dynamic height: ~35px per row + header, capped so it scrolls
                _dyn_h = min(max(len(setup_df) * 36 + 40, 200), 600)
                st.dataframe(
                    setup_df, hide_index=True, height=_dyn_h,
                    use_container_width=True, row_height=35,
                    column_config={
                        "Stock":  st.column_config.TextColumn("Stock", width="small", pinned=True),
                        "Action": st.column_config.TextColumn("Action", width="small"),
                        "Grade":  st.column_config.TextColumn("Grade", width="small"),
                        "CMP":    st.column_config.NumberColumn("CMP", format="₹%.2f"),
                        "Entry":  st.column_config.NumberColumn("Entry", format="₹%.2f"),
                        "Target": st.column_config.NumberColumn("Target", format="₹%.2f"),
                        "SL":     st.column_config.NumberColumn("SL", format="₹%.2f"),
                        "RR":     st.column_config.NumberColumn("R:R", format="%.2f"),
                        "SMC":    st.column_config.NumberColumn("Score", format="%d"),
                    })
                st.download_button(
                    "⬇️ Export SMC Setups CSV",
                    setup_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"smc_setups_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv")
            else:
                st.info("No setups found at this quality/action filter. Try lowering to grade B or 'All' actions.")
        else:
            st.info("Click **🎯 Scan Setups** to find SMC trade setups across the universe.")

# ── ETF Tracker ──────────────────────────────────────────────────────────────
elif _page == 'etfs':
    if not _FUNDS_AVAILABLE:
        st.warning("📈 ETF Tracker requires **funds.py** in your repo. "
                   "Upload funds.py from the project outputs to enable this page.",
                   icon="⚠️")
    else:
        st.markdown('<div class="sec">📈 NSE ETF Tracker — Live Prices & Signals</div>',
                    unsafe_allow_html=True)
        st.caption("Curated NSE-listed ETFs across equity, gold, silver, international "
                   "and debt. Live prices via Yahoo Finance with RSI/trend signals.")

        etf_cats = _funds.get_etf_categories()
        ec1, ec2, ec3 = st.columns([1.5, 1.5, 1])
        with ec1:
            sel_cat = st.selectbox("Category", ["All Categories"] + etf_cats,
                                   label_visibility="collapsed")
        with ec2:
            etf_search = st.text_input("Search ETF", placeholder="Search name or symbol…",
                                       label_visibility="collapsed")
        with ec3:
            run_etf_scan = st.button("📊 Scan ETFs", width="stretch")

        if run_etf_scan:
            cat_arg = None if sel_cat == "All Categories" else sel_cat
            with st.spinner("Fetching live ETF data…"):
                st.session_state.etf_scan_cache = _funds.scan_etfs(category=cat_arg)

        etf_data = st.session_state.etf_scan_cache
        if etf_data is None:
            st.info("💡 Click **📊 Scan ETFs** to fetch live prices, returns and signals "
                    "for all curated NSE ETFs.")
        elif etf_data.empty:
            st.warning("No ETF data returned — Yahoo may be rate-limited. Try again shortly.")
        else:
            display_etf = etf_data.copy()
            if sel_cat != "All Categories":
                display_etf = display_etf[display_etf["Category"] == sel_cat]
            if etf_search.strip():
                q = etf_search.strip().upper()
                display_etf = display_etf[
                    display_etf["Symbol"].str.upper().str.contains(q) |
                    display_etf["Name"].str.upper().str.contains(q)]

            # Summary chips per category
            chip_html = ""
            for cat in etf_data["Category"].unique():
                n = len(etf_data[etf_data["Category"] == cat])
                chip_html += (f'<span style="background:var(--card2);border:1px solid '
                              f'var(--border);border-radius:6px;padding:.25rem .6rem;'
                              f'font-size:.72rem;font-weight:700;color:var(--text);'
                              f'margin-right:.3rem">{cat} <b>{n}</b></span>')
            st.markdown(f'<div style="margin-bottom:.8rem">{chip_html}</div>',
                        unsafe_allow_html=True)

            _etf_h = min(max(len(display_etf) * 36 + 40, 200), 620)
            st.dataframe(
                display_etf, hide_index=True, height=_etf_h, use_container_width=True,
                column_config={
                    "Symbol":   st.column_config.TextColumn("Symbol", width="small", pinned=True),
                    "Name":     st.column_config.TextColumn("Name", width="medium"),
                    "Category": st.column_config.TextColumn("Category", width="small"),
                    "CMP":      st.column_config.NumberColumn("CMP", format="₹%.2f"),
                    "1D %":     st.column_config.NumberColumn("1D %", format="%.2f"),
                    "1W %":     st.column_config.NumberColumn("1W %", format="%.2f"),
                    "1M %":     st.column_config.NumberColumn("1M %", format="%.2f"),
                    "3M %":     st.column_config.NumberColumn("3M %", format="%.2f"),
                    "1Y %":     st.column_config.NumberColumn("1Y %", format="%.2f"),
                    "RSI":      st.column_config.NumberColumn("RSI", format="%.1f"),
                    "Vol(₹Cr)": st.column_config.NumberColumn("Vol ₹Cr", format="%.1f"),
                })
            st.download_button(
                "⬇️ Export ETF Data CSV",
                display_etf.to_csv(index=False).encode("utf-8"),
                file_name=f"etf_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv")

# ── Mutual Funds ─────────────────────────────────────────────────────────────
elif _page == 'mutual_funds':
    if not _FUNDS_AVAILABLE:
        st.warning("🏛 Mutual Funds requires **funds.py** in your repo. "
                   "Upload funds.py from the project outputs to enable this page.",
                   icon="⚠️")
    else:
        st.markdown('<div class="sec">🏛 Mutual Fund Explorer — NAV, Returns & SIP</div>',
                    unsafe_allow_html=True)
        st.caption("Search 40,000+ Indian mutual funds via the free MFAPI (AMFI data). "
                   "NAV history, trailing returns, CAGR and SIP calculator.")

        mf_t1, mf_t2, mf_t3 = st.tabs(["🔍 Search & Analyse", "⚖️ Compare Funds", "🧮 SIP Calculator"])

        # ── Search & Analyse ───────────────────────────────────────────────────
        with mf_t1:
            ms1, ms2 = st.columns([3, 1])
            with ms1:
                mf_query = st.text_input(
                    "Fund search", placeholder="e.g. Parag Parikh Flexi, HDFC Mid Cap, Nippon Small…",
                    label_visibility="collapsed")
            with ms2:
                mf_go = st.button("🔍 Search Funds", width="stretch")

            if mf_go and mf_query.strip():
                with st.spinner("Searching AMFI database…"):
                    st.session_state.mf_search_results = _funds.search_mutual_funds(
                        mf_query.strip())
                    st.session_state.mf_selected = None

            results = st.session_state.mf_search_results
            if results:
                st.markdown(
                    f'<div style="font-size:.78rem;color:var(--muted);margin:.4rem 0">'
                    f'{len(results)} funds found — select one to analyse</div>',
                    unsafe_allow_html=True)
                fund_names = [f"{r['schemeName']}" for r in results[:50]]
                sel_fund_name = st.selectbox("Select fund", fund_names,
                                             label_visibility="collapsed")
                sel_fund = next((r for r in results
                                 if r["schemeName"] == sel_fund_name), None)

                if sel_fund and st.button("📊 Analyse This Fund", width="stretch"):
                    with st.spinner("Fetching NAV history & computing returns…"):
                        st.session_state.mf_selected = _funds.get_fund_details(
                            sel_fund["schemeCode"])

            fund = st.session_state.mf_selected
            if fund and not fund.get("error"):
                meta = fund["meta"]
                st.markdown(
                    f'<div style="background:var(--card);border:1px solid var(--accent);'
                    f'border-radius:12px;padding:1.2rem;margin:1rem 0">'
                    f'<div style="font-size:1.05rem;font-weight:800;color:var(--text)">'
                    f'{meta.get("scheme_name","—")}</div>'
                    f'<div style="font-size:.78rem;color:var(--muted);margin-top:.3rem">'
                    f'{meta.get("fund_house","—")} · {meta.get("scheme_category","—")} · '
                    f'Code {meta.get("scheme_code","—")}</div>'
                    f'<div style="font-size:1.6rem;font-weight:800;color:var(--accent);'
                    f'margin-top:.5rem">₹{fund["latest_nav"]:,.4f} '
                    f'<span style="font-size:.75rem;color:var(--muted);font-weight:600">'
                    f'NAV · {fund["latest_date"]}</span></div>'
                    f'</div>',
                    unsafe_allow_html=True)

                # Returns cards
                ret = fund["returns"]
                ret_html = '<div class="cards">'
                for label in ["1M", "3M", "6M", "1Y", "3Y", "5Y"]:
                    v = ret.get(label)
                    if v is None:
                        ret_html += card(label, "—")
                    else:
                        ret_html += card(label, f"{v:+.1f}%", "",
                                         "green" if v >= 0 else "red")
                cagr3 = ret.get("3Y CAGR"); cagr5 = ret.get("5Y CAGR")
                if cagr3 is not None:
                    ret_html += card("3Y CAGR", f"{cagr3:.1f}%", "",
                                     "green" if cagr3 >= 0 else "red")
                if cagr5 is not None:
                    ret_html += card("5Y CAGR", f"{cagr5:.1f}%", "",
                                     "green" if cagr5 >= 0 else "red")
                ret_html += '</div>'
                st.markdown(ret_html, unsafe_allow_html=True)

                # NAV chart
                nav_df = fund.get("nav_history")
                if nav_df is not None and not nav_df.empty:
                    nfig = go.Figure(go.Scatter(
                        x=nav_df["date"], y=nav_df["nav"],
                        line=dict(color=theme_t["accent"], width=2),
                        fill="tozeroy",
                        fillcolor="rgba(212,175,55,0.06)"))
                    nfig.update_layout(
                        title="NAV History (5Y)", height=320,
                        margin=dict(l=10, r=10, t=40, b=10),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color=theme_t.get("text", "#fff")))
                    nfig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
                    nfig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", tickprefix="₹")
                    st.plotly_chart(nfig, use_container_width=True)

                if st.button("➕ Add to Compare List"):
                    entry = {"code": meta.get("scheme_code"),
                             "name": meta.get("scheme_name")}
                    if entry not in st.session_state.mf_compare_list:
                        st.session_state.mf_compare_list.append(entry)
                        st.toast(f"Added to compare ({len(st.session_state.mf_compare_list)})")
            elif fund and fund.get("error"):
                st.error(f"Could not fetch fund details: {fund['error']}")
            elif not results:
                st.info("💡 Search any Indian mutual fund by name — e.g. 'Parag Parikh', "
                        "'Quant Small Cap', 'HDFC Index'.")

        # ── Compare Funds ──────────────────────────────────────────────────────
        with mf_t2:
            clist = st.session_state.mf_compare_list
            if not clist:
                st.info("Add funds to compare from the Search tab (➕ Add to Compare List).")
            else:
                st.markdown(
                    f'<div style="font-size:.8rem;color:var(--muted);margin-bottom:.6rem">'
                    f'Comparing {len(clist)} funds</div>', unsafe_allow_html=True)
                if st.button("🗑 Clear Compare List"):
                    st.session_state.mf_compare_list = []
                    st.rerun()
                if st.button("⚖️ Run Comparison", width="stretch"):
                    with st.spinner("Fetching all funds…"):
                        comp_df = _funds.compare_funds(
                            [c["code"] for c in clist])
                    if comp_df is not None and not comp_df.empty:
                        st.dataframe(
                            comp_df, hide_index=True, use_container_width=True,
                            column_config={
                                "Fund": st.column_config.TextColumn("Fund", width="large"),
                                "NAV":  st.column_config.NumberColumn("NAV", format="₹%.2f"),
                            })
                    else:
                        st.warning("Comparison failed — try again.")

        # ── SIP Calculator ─────────────────────────────────────────────────────
        with mf_t3:
            sp1, sp2, sp3 = st.columns(3)
            with sp1:
                sip_amt = st.number_input("Monthly SIP ₹", min_value=500,
                                          value=10000, step=500)
            with sp2:
                sip_yrs = st.number_input("Years", min_value=1, value=10, step=1)
            with sp3:
                sip_ret = st.number_input("Expected annual return %", min_value=1.0,
                                          value=12.0, step=0.5)
            res = _funds.sip_calculator(sip_amt, sip_yrs, sip_ret)
            st.markdown(
                '<div class="cards">'
                + card("Invested", fi(res["invested"]))
                + card("Est. Value", fi(res["future_value"]), "", "green")
                + card("Wealth Gain", fi(res["gain"]),
                       f'{res["gain_pct"]:.0f}% growth', "green")
                + '</div>',
                unsafe_allow_html=True)
            st.caption("Assumes monthly compounding at the expected return. "
                       "Actual returns vary — this is an illustration, not a projection.")

# ── Position Sizing Calculator ───────────────────────────────────────────────
elif _page == 'sizing':
    st.markdown('<div class="sec">🧮 Position Sizing Calculator</div>',
                unsafe_allow_html=True)
    st.caption("Risk-based position sizing: never risk more than a fixed % of capital "
               "on a single trade. The golden rule of surviving as a trader.")

    pz1, pz2 = st.columns([1, 1])
    with pz1:
        st.markdown('<div style="font-size:.85rem;font-weight:800;margin-bottom:.5rem">'
                    '💰 Capital & Risk</div>', unsafe_allow_html=True)
        cap_total = st.number_input("Total Trading Capital ₹", min_value=1000.0,
                                    value=float(st.session_state.get("_sz_cap", 100000.0)),
                                    step=5000.0, format="%.0f")
        st.session_state._sz_cap = cap_total
        risk_pct = st.slider("Risk per trade (%)", 0.25, 5.0, 1.0, 0.25)
        max_risk = cap_total * risk_pct / 100

    with pz2:
        st.markdown('<div style="font-size:.85rem;font-weight:800;margin-bottom:.5rem">'
                    '🎯 Trade Setup</div>', unsafe_allow_html=True)
        entry_p = st.number_input("Entry Price ₹", min_value=0.01, value=100.0,
                                  step=0.5, format="%.2f")
        sl_p = st.number_input("Stop Loss ₹", min_value=0.01, value=95.0,
                               step=0.5, format="%.2f")
        tgt_p = st.number_input("Target ₹ (optional)", min_value=0.0, value=110.0,
                                step=0.5, format="%.2f")

    if sl_p >= entry_p:
        st.error("Stop loss must be BELOW entry price for a long trade.")
    else:
        risk_per_share = entry_p - sl_p
        qty = int(max_risk // risk_per_share) if risk_per_share > 0 else 0
        position_value = qty * entry_p
        pos_pct_capital = position_value / cap_total * 100 if cap_total > 0 else 0
        actual_risk = qty * risk_per_share
        rr = (tgt_p - entry_p) / risk_per_share if (tgt_p > entry_p and risk_per_share > 0) else None
        potential_gain = qty * (tgt_p - entry_p) if tgt_p > entry_p else 0

        st.markdown(
            '<div class="cards">'
            + card("Max Risk", fi(max_risk), f"{risk_pct}% of capital", "yellow")
            + card("Quantity", f"{qty:,}", f"₹{risk_per_share:.2f} risk/share", "blue")
            + card("Position Size", fi(position_value),
                   f"{pos_pct_capital:.1f}% of capital",
                   "red" if pos_pct_capital > 25 else "")
            + card("Actual Risk", fi(actual_risk), "if SL hits", "red")
            + (card("R:R Ratio", f"{rr:.2f}", f"Gain: {fi(potential_gain)}",
                    "green" if rr and rr >= 2 else "yellow") if rr else "")
            + '</div>',
            unsafe_allow_html=True)

        if pos_pct_capital > 25:
            st.warning(f"⚠️ Position is {pos_pct_capital:.0f}% of capital — consider "
                       "reducing. Concentration kills accounts faster than bad picks.")
        if rr and rr < 1.5:
            st.warning(f"⚠️ R:R of {rr:.2f} is below 1.5 — this trade needs a "
                       ">{100/(1+rr):.0f}% win rate just to break even.")
        elif rr and rr >= 2:
            st.success(f"✅ R:R of {rr:.2f} — at this ratio you only need a "
                       f"{100/(1+rr):.0f}% win rate to be profitable.")

# ── Risk Dashboard ───────────────────────────────────────────────────────────
elif _page == 'risk':
    st.markdown('<div class="sec">🛡 Portfolio Risk Dashboard</div>',
                unsafe_allow_html=True)

    if df.empty or odf.empty:
        st.info("Risk dashboard requires open positions.")
    else:
        # ── Concentration analysis ───────────────────────────────────────────
        conc = odf.groupby("stock")["invested"].sum().sort_values(ascending=False)
        total_inv = conc.sum()
        top1_pct = conc.iloc[0] / total_inv * 100 if total_inv > 0 else 0
        top3_pct = conc.head(3).sum() / total_inv * 100 if total_inv > 0 else 0

        # ── Sector exposure ──────────────────────────────────────────────────
        odf_s = odf.copy()
        odf_s["sector"] = odf_s["stock"].apply(get_sector)
        sec_exp = odf_s.groupby("sector")["invested"].sum().sort_values(ascending=False)
        top_sec_pct = sec_exp.iloc[0] / total_inv * 100 if total_inv > 0 else 0

        # ── Unrealized drawdown per position ─────────────────────────────────
        worst_pos = odf.nsmallest(3, "profit_pct")[["stock", "profit_pct", "profit"]]

        risk_score = 0
        risk_notes = []
        if top1_pct > 30:
            risk_score += 30
            risk_notes.append(f"🔴 Single-stock concentration: {conc.index[0]} is "
                              f"{top1_pct:.0f}% of portfolio")
        elif top1_pct > 20:
            risk_score += 15
            risk_notes.append(f"🟡 {conc.index[0]} is {top1_pct:.0f}% of portfolio")
        if top_sec_pct > 40:
            risk_score += 25
            risk_notes.append(f"🔴 Sector concentration: {sec_exp.index[0]} is "
                              f"{top_sec_pct:.0f}% of portfolio")
        elif top_sec_pct > 30:
            risk_score += 12
            risk_notes.append(f"🟡 {sec_exp.index[0]} sector is {top_sec_pct:.0f}%")
        deep_loss = odf[odf["profit_pct"] < -10]
        if not deep_loss.empty:
            risk_score += 20
            risk_notes.append(f"🔴 {len(deep_loss)} position(s) down >10% — "
                              "review stop discipline")
        n_pos = len(odf["stock"].unique())
        if n_pos < 3:
            risk_score += 15
            risk_notes.append(f"🟡 Only {n_pos} position(s) — low diversification")
        elif n_pos > 15:
            risk_score += 10
            risk_notes.append(f"🟡 {n_pos} positions — may be over-diversified to track")

        regime_now = market.get("regime", "Unknown")
        if regime_now in ("Bear", "Strong Bear"):
            risk_score += 15
            risk_notes.append(f"🔴 Market regime is {regime_now} — elevated systemic risk")

        risk_score = min(risk_score, 100)
        r_clr = ("#10b981" if risk_score < 30 else
                 "#f59e0b" if risk_score < 60 else "#ef4444")
        r_label = ("LOW" if risk_score < 30 else
                   "MODERATE" if risk_score < 60 else "HIGH")

        st.markdown(
            f'<div style="display:flex;gap:1.5rem;align-items:center;flex-wrap:wrap;'
            f'background:var(--card);border:1px solid {r_clr};border-radius:14px;'
            f'padding:1.3rem 1.6rem;margin-bottom:1.5rem">'
            f'<div><div style="font-size:.7rem;color:var(--muted);font-weight:700;'
            f'text-transform:uppercase;letter-spacing:.1em">Portfolio Risk Score</div>'
            f'<div style="font-size:2.4rem;font-weight:800;color:{r_clr}">'
            f'{risk_score}<span style="font-size:1rem;color:var(--muted)">/100</span> '
            f'<span style="font-size:1rem;color:{r_clr}">{r_label}</span></div></div>'
            f'<div style="flex:1;min-width:220px">'
            f'<div style="height:10px;background:var(--input);border-radius:5px">'
            f'<div style="height:10px;border-radius:5px;background:{r_clr};'
            f'width:{risk_score}%"></div></div></div>'
            f'</div>',
            unsafe_allow_html=True)

        if risk_notes:
            notes_html = "".join(
                f'<div style="padding:.5rem .8rem;font-size:.85rem;'
                f'border-left:3px solid var(--border);margin-bottom:.4rem;'
                f'background:var(--card2);border-radius:0 6px 6px 0">{n}</div>'
                for n in risk_notes)
            st.markdown(notes_html, unsafe_allow_html=True)
        else:
            st.success("✅ No major risk flags — concentration, sector exposure "
                       "and drawdowns all within healthy bounds.")

        rk1, rk2 = st.columns(2)
        with rk1:
            st.markdown('<div style="font-size:.85rem;font-weight:800;margin:.8rem 0 .4rem">'
                        '🏗 Stock Concentration</div>', unsafe_allow_html=True)
            cfig = go.Figure(go.Bar(
                x=conc.values, y=conc.index, orientation="h",
                marker_color=[theme_t["red"] if v/total_inv > 0.25
                              else theme_t["accent"] for v in conc.values]))
            cfig.update_layout(
                height=max(200, len(conc)*35 + 60),
                margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=theme_t.get("text", "#fff")))
            cfig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", tickprefix="₹")
            st.plotly_chart(cfig, use_container_width=True)

        with rk2:
            st.markdown('<div style="font-size:.85rem;font-weight:800;margin:.8rem 0 .4rem">'
                        '🏭 Sector Exposure</div>', unsafe_allow_html=True)
            sfig = go.Figure(go.Pie(
                labels=sec_exp.index, values=sec_exp.values, hole=.55,
                marker=dict(colors=px.colors.qualitative.Dark24)))
            sfig.update_layout(
                height=max(200, len(conc)*35 + 60),
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=theme_t.get("text", "#fff")))
            st.plotly_chart(sfig, use_container_width=True)

        if not worst_pos.empty:
            st.markdown('<div style="font-size:.85rem;font-weight:800;margin:.8rem 0 .4rem">'
                        '📉 Weakest Open Positions</div>', unsafe_allow_html=True)
            wp = worst_pos.copy()
            wp.columns = ["Stock", "P&L %", "P&L ₹"]
            st.dataframe(wp, hide_index=True, use_container_width=True)

# ── Price Alerts ─────────────────────────────────────────────────────────────
elif _page == 'alerts':
    st.markdown('<div class="sec">🔔 Price Alerts</div>', unsafe_allow_html=True)
    st.caption("Set above/below price triggers on any NSE stock. Alerts are checked "
               "each time the dashboard refreshes (~5 min) and can push to Telegram.")

    with st.form("alert_form", clear_on_submit=True):
        al1, al2, al3, al4 = st.columns([2, 1, 1.2, 1])
        with al1:
            al_stock = st.text_input("Stock", placeholder="e.g. RELIANCE",
                                     label_visibility="collapsed")
        with al2:
            al_cond = st.selectbox("Condition", ["above", "below"],
                                   label_visibility="collapsed")
        with al3:
            al_price = st.number_input("Price ₹", min_value=0.01, step=0.5,
                                       value=100.0, format="%.2f",
                                       label_visibility="collapsed")
        with al4:
            al_submit = st.form_submit_button("➕ Set Alert", width="stretch")
        al_note = st.text_input("Note (optional)", placeholder="e.g. breakout level / support retest",
                                label_visibility="collapsed")
        if al_submit and al_stock.strip():
            add_price_alert(UID, al_stock, al_cond, al_price, al_note)
            st.toast(f"🔔 Alert set: {al_stock.upper()} {al_cond} ₹{al_price}")
            st.rerun()

    # ── Check active alerts against live prices ────────────────────────────────
    active_alerts = get_price_alerts(UID, status="Active")
    if active_alerts:
        alert_syms = tuple(sorted({a[1] for a in active_alerts}))
        live_p = _cached_prices(alert_syms)
        triggered_now = []
        for aid, stock, cond, tprice, status, note, cdate, tdate in active_alerts:
            lp = live_p.get(stock)
            if lp is None:
                continue
            if (cond == "above" and lp >= tprice) or (cond == "below" and lp <= tprice):
                trigger_price_alert(aid, UID)
                triggered_now.append((stock, cond, tprice, lp))
        if triggered_now:
            for stock, cond, tprice, lp in triggered_now:
                st.toast(f"🚨 {stock} is {cond} ₹{tprice} (now ₹{lp})", icon="🔔")
            # Telegram push if configured
            if saved_tok and saved_cid:
                try:
                    msg = "🔔 <b>PRICE ALERTS TRIGGERED</b>\n" + "\n".join(
                        f"• {s} crossed {c} ₹{t} — now ₹{l}"
                        for s, c, t, l in triggered_now)
                    send_telegram(saved_tok, saved_cid, msg)
                except Exception:
                    pass
            active_alerts = get_price_alerts(UID, status="Active")

    # ── Render active alerts ───────────────────────────────────────────────────
    st.markdown('<div class="sec" style="margin-top:1rem">⏳ Active Alerts</div>',
                unsafe_allow_html=True)
    if not active_alerts:
        st.info("No active alerts. Set one above.")
    else:
        alert_syms = tuple(sorted({a[1] for a in active_alerts}))
        live_p = _cached_prices(alert_syms)
        rows = ""
        for aid, stock, cond, tprice, status, note, cdate, tdate in active_alerts:
            lp = live_p.get(stock)
            lp_str = f"₹{lp:,.2f}" if lp else "—"
            if lp:
                dist = ((tprice - lp) / lp * 100) if cond == "above" else ((lp - tprice) / lp * 100)
                dist_str = f"{dist:+.1f}% away"
                dist_clr = "var(--yellow)" if abs(dist) < 3 else "var(--muted)"
            else:
                dist_str, dist_clr = "—", "var(--muted)"
            cond_badge = ("🔼" if cond == "above" else "🔽")
            rows += (
                f"<tr>"
                f"<td style='text-align:left;font-weight:800'>{stock}</td>"
                f"<td>{cond_badge} {cond} ₹{tprice:,.2f}</td>"
                f"<td>{lp_str}</td>"
                f"<td style='color:{dist_clr};font-weight:600'>{dist_str}</td>"
                f"<td style='text-align:left;font-size:.78rem;color:var(--muted)'>{note or '—'}</td>"
                f"<td style='font-size:.75rem;color:var(--muted)'>{cdate}</td>"
                f"</tr>"
            )
        st.markdown(
            f'<div class="tbl-wrap"><table class="t"><thead><tr>'
            f'<th class="l">Stock</th><th>Trigger</th><th>Live</th>'
            f'<th>Distance</th><th class="l">Note</th><th>Set On</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>',
            unsafe_allow_html=True)

        del_opts = [f"{a[0]} — {a[1]} {a[2]} ₹{a[3]}" for a in active_alerts]
        dc1, dc2 = st.columns([3, 1])
        with dc1:
            sel_del = st.selectbox("Delete alert", del_opts, label_visibility="collapsed")
        with dc2:
            if st.button("🗑 Delete", width="stretch"):
                delete_price_alert(int(sel_del.split(" — ")[0]), UID)
                st.rerun()

    # ── Triggered history ──────────────────────────────────────────────────────
    trig = get_price_alerts(UID, status="Triggered")
    if trig:
        with st.expander(f"✅ Triggered History ({len(trig)})"):
            for aid, stock, cond, tprice, status, note, cdate, tdate in trig[:20]:
                st.markdown(
                    f'<div style="font-size:.82rem;padding:.3rem 0;color:var(--muted)">'
                    f'🔔 <b style="color:var(--text)">{stock}</b> crossed {cond} '
                    f'₹{tprice:,.2f} on {tdate}'
                    f'{(" · " + note) if note else ""}</div>',
                    unsafe_allow_html=True)

# ── Stock Chart ──────────────────────────────────────────────────────────────
elif _page == 'chart':
    st.markdown('<div class="sec">📈 Interactive Stock Chart</div>',
                unsafe_allow_html=True)

    open_syms = (raw[raw["status"]=="Open"]["stock"].unique().tolist()
                 if not raw.empty else [])
    ch1, ch2, ch3 = st.columns([2, 1.5, 1.2])
    with ch1:
        chart_options = open_syms + ["— Custom —"]
        ch_sel = st.selectbox("Stock", chart_options, label_visibility="collapsed")
    with ch2:
        ch_custom = st.text_input("Custom symbol", placeholder="e.g. TCS",
                                  label_visibility="collapsed")
    with ch3:
        ch_tf = st.selectbox("Timeframe", list(_CHART_TIMEFRAMES.keys()),
                             index=3, label_visibility="collapsed")

    ch_sym = (ch_custom.strip().upper() if ch_custom.strip()
              else (ch_sel if ch_sel != "— Custom —" else None))

    if not ch_sym:
        st.info("💡 Pick a holding or enter any NSE symbol to render its chart.")
    else:
        period, interval = _CHART_TIMEFRAMES[ch_tf]
        with st.spinner(f"Loading {ch_sym} ({ch_tf})…"):
            cdf_chart = fetch_chart_data(ch_sym, period, interval)

        if cdf_chart is None or cdf_chart.empty:
            st.error(f"No data for {ch_sym} at {ch_tf}. Try a different timeframe "
                     "or verify the symbol.")
        else:
            show_ema = st.checkbox("Show EMA 9/21", value=True)
            fig = go.Figure(go.Candlestick(
                x=cdf_chart.index, open=cdf_chart["Open"], high=cdf_chart["High"],
                low=cdf_chart["Low"], close=cdf_chart["Close"],
                increasing_line_color=theme_t["green"],
                decreasing_line_color=theme_t["red"], name=ch_sym))
            if show_ema and len(cdf_chart) >= 21:
                ema9s  = cdf_chart["Close"].ewm(span=9,  adjust=False).mean()
                ema21s = cdf_chart["Close"].ewm(span=21, adjust=False).mean()
                fig.add_trace(go.Scatter(x=cdf_chart.index, y=ema9s, name="EMA 9",
                                         line=dict(color=theme_t["yellow"], width=1.2)))
                fig.add_trace(go.Scatter(x=cdf_chart.index, y=ema21s, name="EMA 21",
                                         line=dict(color=theme_t["blue"], width=1.2)))

            # Mark buy price if it's a holding
            if ch_sym in open_syms and not raw.empty:
                b_at = raw[(raw["stock"] == ch_sym) & (raw["status"] == "Open")]["buy_at"].mean()
                fig.add_hline(y=b_at, line_dash="dash", line_color=theme_t["accent"],
                              annotation_text=f"Avg Buy ₹{b_at:,.2f}",
                              annotation_font_color=theme_t["accent"])

            fig.update_layout(
                height=560, xaxis_rangeslider_visible=False,
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=theme_t.get("text", "#fff")),
                legend=dict(orientation="h", y=1.02, font=dict(color=theme_t.get("text", "#000"), size=11)))
            _ax_font = dict(color=theme_t.get("text", "#fff"), size=11)
            fig.update_xaxes(gridcolor="rgba(148,163,184,0.18)", tickfont=_ax_font)
            fig.update_yaxes(gridcolor="rgba(148,163,184,0.18)", tickprefix="₹", tickfont=_ax_font)
            st.plotly_chart(fig, use_container_width=True)

            # Volume subchart
            vfig = go.Figure(go.Bar(
                x=cdf_chart.index, y=cdf_chart["Volume"],
                marker_color=[theme_t["green"] if c >= o else theme_t["red"]
                              for c, o in zip(cdf_chart["Close"], cdf_chart["Open"])]))
            vfig.update_layout(
                height=160, margin=dict(l=10, r=10, t=5, b=10), showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=theme_t.get("text", "#fff")))
            vfig.update_xaxes(gridcolor="rgba(148,163,184,0.12)", tickfont=_ax_font)
            vfig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", tickfont=_ax_font)
            st.plotly_chart(vfig, use_container_width=True)

# ── Trade Journal ────────────────────────────────────────────────────────────
elif _page == 'journal':
    st.markdown('<div class="sec">📓 Trade Journal</div>', unsafe_allow_html=True)
    st.caption("The habit that separates professionals from gamblers: log every trade's "
               "setup, reasoning, emotion and lesson. Your edge lives in this data.")

    with st.expander("➕ Log a Trade", expanded=False):
        with st.form("journal_form", clear_on_submit=True):
            j1, j2, j3 = st.columns(3)
            with j1:
                j_stock = st.text_input("Stock *", placeholder="e.g. TATAPOWER")
                j_date = st.date_input("Trade Date")
                j_dir = st.selectbox("Direction", ["Long", "Short"])
            with j2:
                j_entry = st.number_input("Entry ₹", min_value=0.0, step=0.5, format="%.2f")
                j_exit = st.number_input("Exit ₹ (0 if open)", min_value=0.0, step=0.5, format="%.2f")
                j_setup = st.selectbox("Setup", [
                    "Breakout", "Pullback to EMA", "VCP / Base", "Bear Trap Reversal",
                    "SMC Order Block", "Sector Momentum", "Earnings Play",
                    "Support Bounce", "Other"])
            with j3:
                j_emotion = st.selectbox("Emotional state", [
                    "Calm / Planned", "FOMO", "Revenge trade", "Fearful",
                    "Overconfident", "Bored / Impulsive"])
                j_outcome = st.selectbox("Outcome", ["Open", "Win", "Loss", "Breakeven"])
                j_rating = st.slider("Execution rating", 1, 5, 3,
                                     help="Rate the QUALITY of your process, not the P&L")
            j_rationale = st.text_area("Why did you take this trade? *",
                                       placeholder="Setup, confluence, plan…", height=80)
            j_lesson = st.text_area("Lesson / what would you do differently?",
                                    placeholder="Filled after close ideally", height=60)
            if st.form_submit_button("💾 Save Entry", width="stretch"):
                if j_stock.strip() and j_rationale.strip():
                    add_journal_entry(
                        UID, j_stock, j_date.strftime("%Y-%m-%d"), j_dir,
                        j_entry if j_entry > 0 else None,
                        j_exit if j_exit > 0 else None,
                        j_setup, j_rationale, j_emotion, j_outcome,
                        j_lesson, j_rating)
                    st.toast(f"📓 Logged {j_stock.upper()}")
                    st.rerun()
                else:
                    st.error("Stock and rationale are required.")

    entries = get_journal_entries(UID)
    if not entries:
        st.info("No journal entries yet. Log your first trade above — future you "
                "will thank present you.")
    else:
        # ── Journal analytics ──────────────────────────────────────────────────
        jdf = pd.DataFrame(entries, columns=[
            "id", "stock", "trade_date", "direction", "entry_price", "exit_price",
            "setup", "rationale", "emotion", "outcome", "lesson", "rating",
            "created_date"])
        closed_j = jdf[jdf["outcome"].isin(["Win", "Loss", "Breakeven"])]

        if not closed_j.empty:
            n_win = len(closed_j[closed_j["outcome"] == "Win"])
            n_all = len(closed_j)
            win_rt = n_win / n_all * 100 if n_all else 0
            best_setup = (closed_j[closed_j["outcome"] == "Win"]["setup"].mode().iloc[0]
                          if n_win > 0 else "—")
            worst_emotion = (closed_j[closed_j["outcome"] == "Loss"]["emotion"].mode().iloc[0]
                             if len(closed_j[closed_j["outcome"] == "Loss"]) > 0 else "—")
            avg_rating = closed_j["rating"].mean()

            st.markdown(
                '<div class="cards">'
                + card("Journaled Trades", str(len(jdf)))
                + card("Win Rate", f"{win_rt:.0f}%", f"{n_win}/{n_all} closed",
                       "green" if win_rt >= 50 else "red")
                + card("Best Setup", best_setup, "most wins", "green")
                + card("Loss Emotion", worst_emotion, "most common in losses", "red")
                + card("Avg Execution", f"{avg_rating:.1f}/5", "process quality",
                       "green" if avg_rating >= 3.5 else "yellow")
                + '</div>',
                unsafe_allow_html=True)

            # Setup performance table
            setup_perf = closed_j.groupby("setup").agg(
                trades=("id", "count"),
                wins=("outcome", lambda x: (x == "Win").sum())).reset_index()
            setup_perf["win %"] = (setup_perf["wins"] / setup_perf["trades"] * 100).round(0)
            setup_perf = setup_perf.sort_values("win %", ascending=False)
            with st.expander("📊 Win Rate by Setup"):
                st.dataframe(setup_perf, hide_index=True, use_container_width=True)

        # ── Entry cards ────────────────────────────────────────────────────────
        st.markdown('<div class="sec" style="margin-top:1rem">📜 Entries</div>',
                    unsafe_allow_html=True)
        for e in entries[:30]:
            (jid, stock, tdate, jdir, entry_p, exit_p, setup, rationale,
             emotion, outcome, lesson, rating, cdate) = e
            o_clr = {"Win": "var(--green)", "Loss": "var(--red)",
                     "Breakeven": "var(--yellow)"}.get(outcome, "var(--muted)")
            pnl_str = ""
            if entry_p and exit_p:
                pnl_pct = ((exit_p - entry_p) / entry_p * 100 *
                           (1 if jdir == "Long" else -1))
                pnl_str = f' · <b style="color:{o_clr}">{pnl_pct:+.1f}%</b>'
            stars = "⭐" * int(rating or 0)
            with st.expander(f"{stock} · {tdate} · {setup} · {outcome} {stars}"):
                st.markdown(
                    f'<div style="font-size:.85rem;line-height:1.8">'
                    f'<b>{jdir}</b> — Entry ₹{entry_p or "—"} → Exit ₹{exit_p or "open"}'
                    f'{pnl_str}<br>'
                    f'<b>Emotion:</b> {emotion} · <b>Outcome:</b> '
                    f'<span style="color:{o_clr};font-weight:700">{outcome}</span><br>'
                    f'<b>Why:</b> {rationale}<br>'
                    f'{("<b>Lesson:</b> " + lesson) if lesson else ""}'
                    f'</div>',
                    unsafe_allow_html=True)
                if st.button("🗑 Delete entry", key=f"jdel_{jid}"):
                    delete_journal_entry(jid, UID)
                    st.rerun()

# ── Market Breadth ───────────────────────────────────────────────────────────
elif _page == 'news':
    st.markdown('<div class="sec">📰 Market News — Live Indian Market Feed</div>',
                unsafe_allow_html=True)
    st.caption("Multi-source feed across the whole market — not just your holdings. "
               "Primary publishers (Business Standard, Livemint, Economic Times, "
               "Moneycontrol), de-duplicated and newest-first.")

    if not _MNEWS_AVAILABLE:
        st.warning("`market_news.py` isn't in the repo yet — upload it to enable "
                   "the market-wide news feed.", icon="⚠️")
    else:
        _c1, _c2 = st.columns([3, 1])
        with _c1:
            _cats = st.multiselect(
                "Categories",
                _mnews.CATEGORIES,
                default=["Markets", "Results", "Economy", "IPO"],
                help="Markets = indices/stocks · Results = earnings · "
                     "Economy = RBI/policy/macro · Global = overnight cues")
        with _c2:
            st.write("")
            _refresh = st.button("🔄 Refresh", use_container_width=True)

        if _refresh:
            _mnews._CACHE.update({"ts": 0, "items": None, "key": None})

        _f1, _f2 = st.columns(2)
        with _f1:
            _sent_filter = st.radio(
                "Sentiment", ["All", "🟢 Bullish only", "🔴 Bearish only"],
                horizontal=True, label_visibility="collapsed")
        with _f2:
            _stock_filter = st.text_input(
                "Filter by stock", placeholder="Filter by stock symbol (e.g. RELIANCE)",
                label_visibility="collapsed").strip().upper()

        with st.spinner("Fetching latest market news…"):
            try:
                _news = _mnews.fetch_market_news(
                    categories=_cats or None, max_total=60)
            except Exception as _e:
                _news = []
                st.error(f"News fetch failed: {_e}")

        # Tag each story with the stocks it mentions + a finance sentiment score
        if _news and _NANA_AVAILABLE:
            try:
                _nana.enrich_news(_news)
                _summary = _nana.sentiment_summary(_news)
            except Exception:
                _summary = None
        else:
            _summary = None

        # Apply user filters (summary above is computed on the UNFILTERED set so
        # the market-wide gauge stays honest)
        if _news and _sent_filter != "All":
            _want = "Bullish" if "Bullish" in _sent_filter else "Bearish"
            _news = [i for i in _news
                     if (i.get("sentiment") or {}).get("label") == _want]
        if _news and _stock_filter:
            _news = [i for i in _news
                     if _stock_filter in (i.get("stocks") or [])]

        if not _news:
            st.info("No stories match those filters (or feeds are unreachable — "
                    "check the feed health panel below).")
        else:
            # ── News-flow sentiment gauge ────────────────────────────────────
            if _summary:
                _b, _r, _n = _summary["bullish"], _summary["bearish"], _summary["neutral"]
                _tot = max(_summary["total"], 1)
                _m1, _m2, _m3 = st.columns(3)
                _m1.metric("🟢 Bullish stories", _b, f"{_b/_tot*100:.0f}%")
                _m2.metric("🔴 Bearish stories", _r, f"{_r/_tot*100:.0f}%")
                _m3.metric("⚪ Neutral / unclear", _n, f"{_n/_tot*100:.0f}%")
                st.caption("⚠️ Sentiment is a keyword-based estimate, not a language "
                           "model — it can misread negation and nuance. Use it to "
                           "decide what to READ, not as a trade signal.")

                if _summary["top_stocks"]:
                    with st.expander("📈 Most-mentioned stocks in the news right now"):
                        st.dataframe(pd.DataFrame([{
                            "Stock": t["symbol"],
                            "Mentions": t["mentions"],
                            "Bullish": t["bull"],
                            "Bearish": t["bear"],
                            "Tilt": t["tilt"],
                        } for t in _summary["top_stocks"]]),
                            use_container_width=True, hide_index=True)
                st.markdown("---")

            st.caption(f"{len(_news)} stories · newest first")
            _cat_colors = {
                "Markets": "#3b82f6", "Results": "#8b5cf6", "Companies": "#06b6d4",
                "Economy": "#f59e0b", "IPO": "#10b981", "Commodities": "#ef4444",
                "Global": "#6b7280",
            }
            for _it in _news:
                _col = _cat_colors.get(_it["category"], "#6b7280")
                _title = _it["title"].replace("<", "&lt;").replace(">", "&gt;")
                _link = _it.get("link") or ""
                _title_html = (
                    f'<a href="{_link}" target="_blank" '
                    f'style="color:var(--text);text-decoration:none;font-weight:600">'
                    f'{_title}</a>' if _link else
                    f'<span style="font-weight:600">{_title}</span>')
                # Sentiment badge — only shown when there was real evidence.
                # A "low"/"none" confidence call is NOT displayed as a verdict,
                # because a keyword scorer with one weak word is a coin-flip.
                _sent = _it.get("sentiment") or {}
                _lab, _conf = _sent.get("label"), _sent.get("confidence")
                _sent_html = ""
                if _lab and _lab != "Neutral" and _conf in ("high", "medium"):
                    _sc = "#10b981" if _lab == "Bullish" else "#ef4444"
                    _icon = "▲" if _lab == "Bullish" else "▼"
                    _dim = "" if _conf == "high" else "opacity:.65;"
                    _sent_html = (
                        f'<span title="keyword estimate · {_conf} confidence · '
                        f'{", ".join(_sent.get("hits", [])[:4])}" '
                        f'style="background:{_sc}22;color:{_sc};{_dim}'
                        f'padding:.1rem .45rem;border-radius:4px;font-size:.7rem;'
                        f'font-weight:700;margin-left:.3rem">{_icon} {_lab}</span>')

                # Stocks mentioned — chips
                _chips = ""
                for _s in (_it.get("stocks") or [])[:4]:
                    _chips += (f'<span style="background:var(--border);'
                               f'color:var(--text);padding:.1rem .4rem;'
                               f'border-radius:4px;font-size:.7rem;font-weight:600;'
                               f'margin-right:.25rem">{_s}</span>')
                _cue_html = ""
                for _cu in (_it.get("cues") or [])[:2]:
                    _cue_html += (f'<span style="color:var(--muted);font-size:.7rem;'
                                  f'margin-right:.4rem">🌐 {_cu}</span>')

                _meta2 = ""
                if _chips or _cue_html:
                    _meta2 = (f'<div style="margin-top:.35rem">{_chips}{_cue_html}</div>')

                st.markdown(
                    f'<div style="padding:.7rem 0;border-bottom:1px solid var(--border)">'
                    f'<span style="background:{_col}22;color:{_col};'
                    f'padding:.1rem .5rem;border-radius:4px;font-size:.7rem;'
                    f'font-weight:700;letter-spacing:.03em">{_it["category"].upper()}</span>'
                    f'{_sent_html} '
                    f'{_title_html}'
                    f'{_meta2}'
                    f'<div style="color:var(--muted);font-size:.75rem;margin-top:.25rem">'
                    f'{_it["source"]} · {_it["age"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True)

        # ── Per-stock lookup ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="sec">🔍 News for a Specific Stock</div>',
                    unsafe_allow_html=True)
        _sc1, _sc2 = st.columns([3, 1])
        with _sc1:
            _sym = st.text_input("Symbol", placeholder="e.g. RELIANCE, TCS, GODREJPROP…",
                                 label_visibility="collapsed")
        with _sc2:
            _go = st.button("Search", use_container_width=True)
        if _go and _sym.strip():
            with st.spinner(f"Fetching news for {_sym.upper()}…"):
                _sn = _mnews.fetch_stock_news(_sym.strip())
            if not _sn:
                st.info(f"No recent news found for {_sym.upper()}.")
            else:
                for _it in _sn:
                    _link = _it.get("link") or ""
                    _t = _it["title"].replace("<", "&lt;").replace(">", "&gt;")
                    _th = (f'<a href="{_link}" target="_blank" '
                           f'style="color:var(--text);text-decoration:none;'
                           f'font-weight:600">{_t}</a>' if _link
                           else f'<span style="font-weight:600">{_t}</span>')
                    st.markdown(
                        f'<div style="padding:.6rem 0;border-bottom:1px solid var(--border)">'
                        f'{_th}<div style="color:var(--muted);font-size:.75rem;'
                        f'margin-top:.25rem">{_it["source"]} · {_it["age"]}</div></div>',
                        unsafe_allow_html=True)

        # ── Feed health — spot a source that changed its URL or blocked us ──
        with st.expander("🩺 Feed health (which sources are alive)"):
            st.caption("If a publisher changes its RSS path or starts blocking "
                       "requests, that feed silently disappears. This shows which "
                       "are actually returning stories right now.")
            if st.button("Run feed health check"):
                with st.spinner("Checking all feeds…"):
                    _h = _mnews.feed_health()
                _alive = sum(1 for r in _h if r["ok"])
                st.markdown(f"**{_alive}/{len(_h)} feeds alive**")
                st.dataframe(
                    pd.DataFrame([{
                        "Status": "✅ OK" if r["ok"] else "❌ FAIL",
                        "Category": r["category"],
                        "Source": r["source"],
                        "Items": r["items"],
                    } for r in _h]),
                    use_container_width=True, hide_index=True)

elif _page == 'breadth':
    st.markdown('<div class="sec">📊 Market Breadth — Universe Internals</div>',
                unsafe_allow_html=True)
    st.caption("The market's true health: how many stocks are actually participating. "
               "Breadth divergence often leads price at turning points.")

    if st.button("📊 Compute Breadth", width="stretch"):
        with st.spinner(f"Analysing breadth across {len(SECTOR_MAP)} stocks… (uses cached data when available)"):
            all_syms = list(SECTOR_MAP.keys())
            bulk = _bulk_fetch_history(all_syms, period="6mo")
            up_day = dn_day = 0
            above20 = above50 = above200 = 0
            nh_20 = nl_20 = 0
            total_valid = 0
            adv_vol = dec_vol = 0.0
            for sym, bdf in bulk.items():
                try:
                    if bdf is None or len(bdf) < 50:
                        continue
                    close = bdf["Close"]
                    c_now = float(close.iloc[-1]); c_prev = float(close.iloc[-2])
                    total_valid += 1
                    v_now = float(bdf["Volume"].iloc[-1]) * c_now
                    if c_now > c_prev:
                        up_day += 1; adv_vol += v_now
                    elif c_now < c_prev:
                        dn_day += 1; dec_vol += v_now
                    e20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
                    e50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
                    if c_now > e20: above20 += 1
                    if c_now > e50: above50 += 1
                    if len(close) >= 200:
                        e200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])
                        if c_now > e200: above200 += 1
                    if c_now >= float(close.iloc[-20:].max()) * 0.999: nh_20 += 1
                    if c_now <= float(close.iloc[-20:].min()) * 1.001: nl_20 += 1
                except Exception:
                    continue

            st.session_state["_breadth_cache"] = {
                "up": up_day, "dn": dn_day, "total": total_valid,
                "a20": above20, "a50": above50, "a200": above200,
                "nh": nh_20, "nl": nl_20,
                "adv_vol": adv_vol, "dec_vol": dec_vol,
                "ts": datetime.now().strftime("%H:%M"),
            }

    br = st.session_state.get("_breadth_cache")
    if not br:
        st.info("💡 Click **📊 Compute Breadth** to measure universe participation. "
                "Fast if a deep scan already ran (shares its cache).")
    else:
        tot = max(br["total"], 1)
        adr = br["up"] / max(br["dn"], 1)
        pct20  = br["a20"] / tot * 100
        pct50  = br["a50"] / tot * 100
        pct200 = br["a200"] / tot * 100
        vol_ratio = br["adv_vol"] / max(br["dec_vol"], 1)

        breadth_score = (
            (25 if pct50 > 60 else 12 if pct50 > 45 else 0) +
            (25 if pct20 > 60 else 12 if pct20 > 45 else 0) +
            (25 if adr > 1.5 else 12 if adr > 1.0 else 0) +
            (25 if br["nh"] > br["nl"] * 2 else 12 if br["nh"] > br["nl"] else 0))
        b_clr = ("#10b981" if breadth_score >= 70 else
                 "#f59e0b" if breadth_score >= 40 else "#ef4444")
        b_label = ("HEALTHY" if breadth_score >= 70 else
                   "MIXED" if breadth_score >= 40 else "WEAK")

        st.markdown(
            f'<div style="display:flex;gap:1.5rem;align-items:center;flex-wrap:wrap;'
            f'background:var(--card);border:1px solid {b_clr};border-radius:14px;'
            f'padding:1.3rem 1.6rem;margin-bottom:1.5rem">'
            f'<div><div style="font-size:.7rem;color:var(--muted);font-weight:700;'
            f'text-transform:uppercase;letter-spacing:.1em">Breadth Health</div>'
            f'<div style="font-size:2.4rem;font-weight:800;color:{b_clr}">'
            f'{breadth_score}<span style="font-size:1rem;color:var(--muted)">/100</span> '
            f'<span style="font-size:1rem;color:{b_clr}">{b_label}</span></div></div>'
            f'<div style="font-size:.78rem;color:var(--muted)">'
            f'{br["total"]} stocks analysed · as of {br["ts"]}</div>'
            f'</div>',
            unsafe_allow_html=True)

        st.markdown(
            '<div class="cards">'
            + card("Adv / Dec", f'{br["up"]} / {br["dn"]}',
                   f"ratio {adr:.2f}", "green" if adr > 1 else "red")
            + card("Above 20 EMA", f"{pct20:.0f}%", f'{br["a20"]} stocks',
                   "green" if pct20 > 50 else "red")
            + card("Above 50 EMA", f"{pct50:.0f}%", f'{br["a50"]} stocks',
                   "green" if pct50 > 50 else "red")
            + card("Above 200 EMA", f"{pct200:.0f}%", f'{br["a200"]} stocks',
                   "green" if pct200 > 50 else "red")
            + card("20d Highs / Lows", f'{br["nh"]} / {br["nl"]}', "",
                   "green" if br["nh"] > br["nl"] else "red")
            + card("Up:Down Volume", f"{vol_ratio:.2f}", "₹ weighted",
                   "green" if vol_ratio > 1 else "red")
            + '</div>',
            unsafe_allow_html=True)

        st.markdown(
            f'<div style="background:var(--card2);border-radius:8px;padding:.8rem 1rem;'
            f'font-size:.82rem;color:var(--muted);line-height:1.7">'
            f'💡 <b style="color:var(--text)">Reading breadth:</b> '
            f'A rising Nifty with breadth &lt;40 is a narrow, fragile rally (few large caps '
            f'carrying it). Breadth &gt;70 with new highs expanding confirms broad '
            f'participation — the safest environment for swing longs. '
            f'20d Lows expanding while the index holds up is an early warning.</div>',
            unsafe_allow_html=True)

# ── Custom Screener ──────────────────────────────────────────────────────────
elif _page == 'screener':
    st.markdown('<div class="sec">🔬 Custom Stock Screener</div>',
                unsafe_allow_html=True)
    st.caption("Build your own filter across the loaded universe. Runs on cached "
               "indicator data — instant if a deep scan already completed.")

    scf1, scf2, scf3, scf4 = st.columns(4)
    with scf1:
        f_rsi_min = st.number_input("RSI min", 0, 100, 0)
        f_rsi_max = st.number_input("RSI max", 0, 100, 100)
    with scf2:
        f_trend = st.multiselect("Trend", ["Strong Uptrend", "Uptrend", "Recovery",
                                           "Sideways", "Downtrend", "Strong Downtrend"],
                                 default=[])
        f_supertrend = st.selectbox("Supertrend", ["Any", "Bullish", "Bearish"])
    with scf3:
        f_price_min = st.number_input("Price min ₹", 0.0, value=0.0, step=10.0)
        f_price_max = st.number_input("Price max ₹", 0.0, value=0.0, step=10.0,
                                      help="0 = no max")
    with scf4:
        f_vol_surge = st.checkbox("Volume surge (>1.5x avg)")
        f_macd_bull = st.checkbox("MACD bullish")
        f_bb_squeeze = st.checkbox("BB squeeze")
        f_liquid = st.checkbox("Liquid only (₹1Cr+)", value=True)

    f_sectors = st.multiselect("Sectors (empty = all)",
                               sorted(set(SECTOR_MAP.values())), default=[])

    if st.button("🔬 Run Screen", width="stretch"):
        with st.spinner(f"Screening {len(SECTOR_MAP)} stocks…"):
            syms = [s for s in SECTOR_MAP
                    if not f_sectors or SECTOR_MAP[s] in f_sectors]
            bulk = _bulk_fetch_history(syms, period="6mo")
            hits = []
            for sym in syms:
                try:
                    ind = compute_indicators(sym, period="6mo",
                                             prefetched_df=bulk.get(sym))
                    if not ind:
                        continue
                    rsi_v = ind.get("rsi") or 0
                    cmp_v = ind.get("cmp") or 0
                    if not (f_rsi_min <= rsi_v <= f_rsi_max):
                        continue
                    if f_price_min and cmp_v < f_price_min:
                        continue
                    if f_price_max and cmp_v > f_price_max:
                        continue
                    if f_trend and ind.get("trend") not in f_trend:
                        continue
                    if f_supertrend == "Bullish" and not ind.get("supertrend_bullish"):
                        continue
                    if f_supertrend == "Bearish" and ind.get("supertrend_bullish"):
                        continue
                    if f_vol_surge and (ind.get("vol_ratio") or 0) < 1.5:
                        continue
                    if f_macd_bull and not ind.get("macd_bullish"):
                        continue
                    if f_bb_squeeze and not ind.get("bb_squeeze"):
                        continue
                    if f_liquid and not ind.get("liquidity_ok", True):
                        continue
                    hits.append({
                        "Stock": sym, "Sector": SECTOR_MAP[sym],
                        "CMP": cmp_v, "RSI": rsi_v,
                        "Trend": ind.get("trend", "—"),
                        "ST": "🟢" if ind.get("supertrend_bullish") else "🔴",
                        "MACD": "🟢" if ind.get("macd_bullish") else "—",
                        "Vol x": round(ind.get("vol_ratio") or 0, 1),
                        "Support": ind.get("support"),
                        "Resist": ind.get("resistance"),
                    })
                except Exception:
                    continue
            st.session_state["_screener_hits"] = hits

    hits = st.session_state.get("_screener_hits")
    if hits is None:
        st.info("Set your filters and click **🔬 Run Screen**.")
    elif not hits:
        st.warning("0 stocks matched. Loosen the filters.")
    else:
        hdf = pd.DataFrame(hits)
        st.markdown(
            f'<div style="font-size:.78rem;color:var(--muted);margin:.5rem 0;'
            f'font-weight:600">✅ {len(hdf)} matches</div>',
            unsafe_allow_html=True)
        _sh = min(max(len(hdf) * 36 + 40, 200), 600)
        st.dataframe(
            hdf, hide_index=True, height=_sh, use_container_width=True,
            column_config={
                "Stock":   st.column_config.TextColumn("Stock", width="small", pinned=True),
                "CMP":     st.column_config.NumberColumn("CMP", format="₹%.2f"),
                "RSI":     st.column_config.NumberColumn("RSI", format="%.1f"),
                "Support": st.column_config.NumberColumn("Support", format="₹%.2f"),
                "Resist":  st.column_config.NumberColumn("Resist", format="₹%.2f"),
            })
        st.download_button(
            "⬇️ Export Screen CSV",
            hdf.to_csv(index=False).encode("utf-8"),
            file_name=f"screen_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv")

# ── Earnings Calendar ────────────────────────────────────────────────────────
elif _page == 'earnings':
    st.markdown('<div class="sec">📆 Earnings Calendar — Your Holdings & Watchlist</div>',
                unsafe_allow_html=True)
    st.caption("Upcoming result dates from Yahoo Finance. Earnings = volatility events: "
               "size positions accordingly or step aside.")

    open_syms = (raw[raw["status"]=="Open"]["stock"].unique().tolist()
                 if not raw.empty else [])
    wl_df = get_watchlist(UID)
    wl_syms = wl_df["stock"].tolist() if not wl_df.empty else []
    all_track = sorted(set(open_syms + wl_syms))

    if not all_track:
        st.info("Add holdings or watchlist stocks to see their earnings dates.")
    else:
        if st.button("📆 Fetch Earnings Dates", width="stretch"):
            import yfinance as _yf2
            rows = []
            with st.spinner(f"Checking {len(all_track)} stocks…"):
                for sym in all_track:
                    try:
                        t = _yf2.Ticker(sym + ".NS")
                        cal = t.calendar
                        e_date = None
                        if cal is not None:
                            if isinstance(cal, dict):
                                ed = cal.get("Earnings Date")
                                if ed:
                                    e_date = ed[0] if isinstance(ed, (list, tuple)) else ed
                            elif hasattr(cal, "empty") and not cal.empty:
                                if "Earnings Date" in cal.index:
                                    e_date = cal.loc["Earnings Date"].iloc[0]
                        if e_date is not None:
                            e_ts = pd.Timestamp(e_date)
                            days = (e_ts - pd.Timestamp.now()).days
                            rows.append({
                                "Stock": sym,
                                "Type": "📌 Holding" if sym in open_syms else "👁 Watchlist",
                                "Earnings Date": e_ts.strftime("%d %b %Y"),
                                "Days Away": days,
                            })
                    except Exception:
                        continue
            st.session_state._earnings_cache = rows

        e_rows = st.session_state.get("_earnings_cache")
        if e_rows is None:
            st.info("Click **📆 Fetch Earnings Dates** to load result dates.")
        elif not e_rows:
            st.warning("No earnings dates found — Yahoo may not have published them yet.")
        else:
            edf = pd.DataFrame(e_rows).sort_values("Days Away")
            imminent = edf[edf["Days Away"].between(0, 7)]
            if not imminent.empty:
                names = ", ".join(imminent["Stock"].tolist())
                st.warning(f"⚡ Results within 7 days: **{names}** — expect volatility; "
                           "review position sizes.", icon="📆")
            st.dataframe(
                edf, hide_index=True, use_container_width=True,
                column_config={
                    "Days Away": st.column_config.NumberColumn("Days", format="%d"),
                })

# ── IPO Tracker ──────────────────────────────────────────────────────────────
elif _page == 'ipo':
    st.markdown('<div class="sec">🆕 Recent IPO Tracker</div>', unsafe_allow_html=True)
    st.caption("Track recently listed NSE stocks — early bases on strong debuts are "
               "classic swing setups (but volatile; size small).")

    with st.form("ipo_form", clear_on_submit=True):
        ip1, ip2 = st.columns([3, 1])
        with ip1:
            ipo_sym = st.text_input("Add recently listed symbol",
                                    placeholder="e.g. SWIGGY, NTPCGREEN",
                                    label_visibility="collapsed")
        with ip2:
            if st.form_submit_button("➕ Track", width="stretch") and ipo_sym.strip():
                s = ipo_sym.strip().upper()
                if s not in st.session_state._ipo_watch:
                    st.session_state._ipo_watch.append(s)

    ipo_list = st.session_state.get("_ipo_watch", [])
    if not ipo_list:
        st.info("Add recently listed symbols above. (Session-only list — clears on restart.)")
    else:
        with st.spinner("Fetching IPO performance…"):
            ipo_bulk = _bulk_fetch_history(ipo_list, period="6mo")
        cards_row = ""
        for sym in ipo_list:
            bdf = ipo_bulk.get(sym)
            if bdf is None or bdf.empty:
                cards_row += (
                    f'<div style="background:var(--card);border:1px solid var(--border);'
                    f'border-radius:10px;padding:1rem;min-width:200px;flex:1">'
                    f'<b>{sym}</b><br><span style="font-size:.78rem;color:var(--red)">'
                    f'No data — verify symbol</span></div>')
                continue
            close = bdf["Close"]
            list_price = float(close.iloc[0])
            cmp_now = float(close.iloc[-1])
            chg = (cmp_now / list_price - 1) * 100
            days_listed = len(close)
            hi = float(close.max()); lo = float(close.min())
            from_high = (cmp_now / hi - 1) * 100
            clr = "var(--green)" if chg >= 0 else "var(--red)"
            cards_row += (
                f'<div style="background:var(--card);border:1px solid var(--border);'
                f'border-radius:10px;padding:1rem;min-width:220px;flex:1">'
                f'<div style="font-weight:800">{sym} '
                f'<span style="font-size:.68rem;color:var(--muted)">'
                f'{days_listed} sessions</span></div>'
                f'<div style="font-size:1.3rem;font-weight:800;color:{clr};margin:.2rem 0">'
                f'{chg:+.1f}%<span style="font-size:.7rem;color:var(--muted)"> since data start</span></div>'
                f'<div style="font-size:.72rem;color:var(--muted);line-height:1.6">'
                f'CMP ₹{cmp_now:,.1f} · Range ₹{lo:,.0f}–₹{hi:,.0f}<br>'
                f'From high: <b style="color:{"var(--red)" if from_high < -15 else "var(--text)"}">'
                f'{from_high:+.1f}%</b></div></div>')
        st.markdown(
            f'<div style="display:flex;gap:.8rem;flex-wrap:wrap">{cards_row}</div>',
            unsafe_allow_html=True)

        if st.button("🗑 Clear IPO list"):
            st.session_state._ipo_watch = []
            st.rerun()

# ── VCP Scanner ──────────────────────────────────────────────────────────────
elif _page == 'vcp':
    st.markdown('<div class="sec">📐 VCP Scanner — Volatility Contraction Patterns</div>',
                unsafe_allow_html=True)
    st.caption("Mark Minervini's signature setup: a base of 2-4 progressively tighter "
               "pullbacks on drying volume, coiling under a pivot. The tighter the "
               "final contraction, the more explosive the breakout tends to be.")

    if scan_for_vcp is None:
        st.warning("📐 VCP Scanner requires the updated **signals.py** (v12+) with "
                   "scan_for_vcp. Deploy the latest signals.py to enable.", icon="⚠️")
    else:
        vc1, vc2, vc3 = st.columns([1.2, 1.2, 1])
        with vc1:
            vcp_min_q = st.selectbox("Min quality", ["C", "B", "A", "A+"], index=1,
                                     label_visibility="collapsed")
        with vc2:
            vcp_ready_only = st.checkbox("🎯 Pivot-ready only", value=False,
                                         help="Only bases within 3% of the buy pivot")
        with vc3:
            run_vcp = st.button("📐 Scan VCP", width="stretch")

        if run_vcp:
            with st.spinner(f"Scanning {len(SECTOR_MAP)} stocks for VCP bases…"):
                st.session_state.vcp_scan_cache = scan_for_vcp(
                    min_quality=vcp_min_q, ready_only=vcp_ready_only)
                vs = st.session_state.vcp_scan_cache
                st.toast(f"✅ {vs['count']} VCP bases · {vs['ready_count']} pivot-ready",
                         icon="📐")

        vs = st.session_state.get("vcp_scan_cache")
        if vs is not None:
            _stale_banner(vs.get("timestamp"), "📐 Scan VCP")
        if vs is None:
            st.info("💡 Click **📐 Scan VCP** to sweep the universe for coiling bases.")
        elif not vs.get("vcp_setups"):
            st.warning("No VCP bases found at this quality filter. Try grade C or "
                       "uncheck pivot-ready.")
        else:
            st.markdown(
                f'<div style="font-size:.75rem;color:var(--muted);margin:.5rem 0">'
                f'Scanned {vs["scanned"]} · {vs["liquid"]} liquid · '
                f'{vs["count"]} bases · {vs["ready_count"]} at pivot · '
                f'scanned at {vs["timestamp"]}</div>',
                unsafe_allow_html=True)
            st.caption("ℹ️ **Close** = last daily close, not a live tick (Yahoo daily "
                       "data, ~15–20 min delayed intraday). VCP pivots, entries and "
                       "targets are all measured on daily closes by design — re-run "
                       "the scan during market hours for the freshest close.")

            vrows = []
            for s in vs["vcp_setups"]:
                _contr = s.get("contractions") or []
                vrows.append({
                    "Stock": s["stock"], "Sector": s["sector"],
                    "Grade": s.get("quality"),
                    "Ready": "🎯" if s.get("vcp_ready") else "",
                    "Close": s.get("cmp"), "Data Date": s.get("data_date"),
                    "Pivot": s.get("pivot"),
                    "To Pivot %": s.get("pivot_distance_pct"),
                    "Contractions": len(_contr),
                    "Sequence": " → ".join(f"-{c}%" for c in _contr) if _contr else "—",
                    "Vol-backed": "🔷" if s.get("vp_backed") else "",
                    "Entry": s.get("entry"),
                    "SL": s.get("stop_loss"), "Target": s.get("target"),
                    "RR": s.get("risk_reward"),
                })
            vdf = pd.DataFrame(vrows)
            _vh = min(max(len(vdf) * 36 + 40, 200), 600)
            st.dataframe(
                vdf, hide_index=True, height=_vh, use_container_width=True,
                column_config={
                    "Stock": st.column_config.TextColumn("Stock", width="small", pinned=True),
                    "Close": st.column_config.NumberColumn("Close", format="₹%.2f",
                        help="Last daily close (not a live tick). VCP pivot/entry/"
                             "targets are all measured on daily closes."),
                    "Pivot": st.column_config.NumberColumn("Pivot", format="₹%.2f"),
                    "To Pivot %": st.column_config.NumberColumn("→Pivot", format="%.1f%%"),
                    "Entry":  st.column_config.NumberColumn("Entry", format="₹%.2f"),
                    "SL":     st.column_config.NumberColumn("SL", format="₹%.2f"),
                    "Target": st.column_config.NumberColumn("Target", format="₹%.2f"),
                    "RR":     st.column_config.NumberColumn("R:R", format="%.2f"),
                })
            st.download_button(
                "⬇️ Export VCP CSV",
                vdf.to_csv(index=False).encode("utf-8"),
                file_name=f"vcp_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv")

            st.markdown(
                '<div style="background:var(--card2);border-radius:8px;padding:.8rem 1rem;'
                'font-size:.8rem;color:var(--muted);line-height:1.7;margin-top:.8rem">'
                '💡 <b style="color:var(--text)">Trading a VCP:</b> buy the pivot break '
                'on volume expansion (ideally 1.5x+ average), stop below the final '
                'contraction low. Grade A+ = 3+ contractions, tight final coil, clear '
                'volume dry-up. Combine with 💪 RS Leaders — a VCP in a leading stock '
                'is the highest-probability setup this engine can find.</div>',
                unsafe_allow_html=True)

# ── RS Leaders ───────────────────────────────────────────────────────────────
elif _page == 'rs':
    st.markdown('<div class="sec">💪 Relative Strength Leaders — vs Nifty 50</div>',
                unsafe_allow_html=True)
    st.caption("IBD-style relative strength: multi-period outperformance vs the index, "
               "percentile-ranked 1-99 across the universe. Leaders lead — buy strength, "
               "not hope.")

    if scan_relative_strength is None:
        st.warning("💪 RS Leaders requires the updated **signals.py** (v12+) with "
                   "scan_relative_strength. Deploy the latest signals.py to enable.",
                   icon="⚠️")
    else:
        rs1, rs2, rs3 = st.columns([1.2, 1.2, 1])
        with rs1:
            rs_min_rating = st.slider("Min RS rating", 50, 99, 80, 5,
                                      label_visibility="collapsed")
        with rs2:
            rs_uptrend_only = st.checkbox("📈 Uptrend only", value=True)
        with rs3:
            run_rs = st.button("💪 Scan RS", width="stretch")

        if run_rs:
            with st.spinner(f"Ranking {len(SECTOR_MAP)} stocks by relative strength…"):
                _rsc_raw = scan_relative_strength(min_rating=rs_min_rating)
                # Uptrend-only filter applied here (signals.py doesn't take the
                # kwarg — we filter its result instead).
                if rs_uptrend_only and _rsc_raw.get("leaders"):
                    _rsc_raw = dict(_rsc_raw)
                    _rsc_raw["leaders"] = [
                        l for l in _rsc_raw["leaders"]
                        if "Uptrend" in str(l.get("trend", ""))]
                    _rsc_raw["count"] = len(_rsc_raw["leaders"])
                st.session_state.rs_scan_cache = _rsc_raw
                rsc = st.session_state.rs_scan_cache
                st.toast(f"✅ {rsc['count']} RS leaders (rating ≥ {rs_min_rating})",
                         icon="💪")

        rsc = st.session_state.get("rs_scan_cache")
        if rsc is None:
            st.info("💡 Click **💪 Scan RS** to rank the universe by relative strength.")
        elif not rsc.get("leaders"):
            st.warning("No stocks passed the RS filter. Lower the minimum rating.")
        else:
            _nifty_ret = rsc.get("nifty_returns") or {}
            _nifty_3m = _nifty_ret.get("63")
            _nifty_3m_str = f'{_nifty_3m:+.1f}%' if isinstance(_nifty_3m, (int, float)) else "—"
            st.markdown(
                f'<div style="font-size:.75rem;color:var(--muted);margin:.5rem 0">'
                f'Scanned {rsc.get("scanned", 0)} · {rsc.get("liquid", 0)} liquid · '
                f'{rsc["count"]} leaders · Nifty 3M: {_nifty_3m_str} · '
                f'{rsc["timestamp"]}</div>',
                unsafe_allow_html=True)

            lrows = []
            for _i, s in enumerate(rsc["leaders"], start=1):
                _r21 = s.get("ret_21d"); _n21 = s.get("nifty_21d")
                _rel_1m = (round(_r21 - _n21, 1)
                           if isinstance(_r21, (int, float)) and isinstance(_n21, (int, float))
                           else None)
                _r63 = s.get("ret_63d")
                _rel_3m = (round(_r63 - _nifty_3m, 1)
                           if isinstance(_r63, (int, float)) and isinstance(_nifty_3m, (int, float))
                           else None)
                lrows.append({
                    "Rank": _i, "Stock": s["stock"], "Sector": s["sector"],
                    "RS Rating": s.get("rs_rating"),
                    "RS Ratio": s.get("rs_ratio"),
                    "1M vs N": _rel_1m,
                    "3M vs N": _rel_3m,
                    "CMP": s.get("cmp"),
                    "Trend": s.get("trend"),
                    "VCP": "📐" if s.get("vcp") else "",
                    "Pivot-Ready": "🎯" if s.get("vcp_ready") else "",
                })
            ldf = pd.DataFrame(lrows)
            _lh = min(max(len(ldf) * 36 + 40, 200), 600)
            st.dataframe(
                ldf, hide_index=True, height=_lh, use_container_width=True,
                column_config={
                    "Rank":  st.column_config.NumberColumn("#", format="%d", width="small"),
                    "Stock": st.column_config.TextColumn("Stock", width="small", pinned=True),
                    "RS Rating": st.column_config.ProgressColumn(
                        "RS", min_value=0, max_value=99, format="%d"),
                    "RS Ratio": st.column_config.NumberColumn("Ratio", format="%.2f"),
                    "1M vs N":  st.column_config.NumberColumn("1M±", format="%+.1f%%"),
                    "3M vs N":  st.column_config.NumberColumn("3M±", format="%+.1f%%"),
                    "CMP":  st.column_config.NumberColumn("CMP", format="₹%.2f"),
                })
            st.download_button(
                "⬇️ Export RS Leaders CSV",
                ldf.to_csv(index=False).encode("utf-8"),
                file_name=f"rs_leaders_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv")

            st.markdown(
                '<div style="background:var(--card2);border-radius:8px;padding:.8rem 1rem;'
                'font-size:.8rem;color:var(--muted);line-height:1.7;margin-top:.8rem">'
                '💡 <b style="color:var(--text)">Using RS:</b> institutions accumulate '
                'leaders — stocks making relative highs before the index confirms. '
                'RS 90+ with a 📐 VCP base and a 🚀 near-high flag is the A+ confluence. '
                'Avoid buying laggards because they "look cheap"; weakness usually '
                'has a reason.</div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# POST-RENDER DEEP SCAN — runs AFTER the page has fully painted.
# Each rerun executes ONE stage, then triggers the next rerun. The user sees
# the dashboard instantly; stages complete in the background between paints.
# Stage order: sector → universe → smc → traps → vcp → rs
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.get("_run_deep_now", False) and st.session_state.get("_deep_running", False):
    _stage = st.session_state.get("_deep_stage", "sector")
    try:
        if _stage == "sector":
            st.session_state.sector_cache = sector_rotation()
            if (st.session_state.sector_cache is not None and
                    not st.session_state.sector_cache.empty):
                st.session_state.outlook_cache = predict_sector_outlook(
                    st.session_state.sector_cache)
                st.session_state.picks_cache = find_sector_picks(
                    st.session_state.sector_cache.head(5)["sector"].tolist(), 3)
            else:
                st.session_state.outlook_cache = pd.DataFrame()
                st.session_state.picks_cache = []
            st.session_state._deep_stage = "universe"

        elif _stage == "universe":
            sd = generate_market_scanner()
            st.session_state.scanner_cache = sd if (sd is not None and not sd.empty) \
                else pd.DataFrame()
            st.session_state._deep_stage = "smc"

        elif _stage == "smc":
            if _SMC_SCANNER_AVAILABLE:
                st.session_state.smc_scan_cache = scan_for_smc_setups(
                    min_quality="B", action_filter="All")
            st.session_state._deep_stage = "traps"

        elif _stage == "traps":
            if _TRAP_SCANNER_AVAILABLE:
                st.session_state.trap_scan_cache = scan_for_traps(min_confidence=60)
            st.session_state._deep_stage = "vcp"

        elif _stage == "vcp":
            if scan_for_vcp is not None:
                st.session_state.vcp_scan_cache = scan_for_vcp(
                    min_quality="B", ready_only=False)
            st.session_state._deep_stage = "rs"

        elif _stage == "rs":
            if scan_relative_strength is not None:
                st.session_state.rs_scan_cache = scan_relative_strength(
                    min_rating=80)
            # Sequence complete
            st.session_state._deep_running = False
            st.session_state._deep_stage = "sector"
            st.session_state.last_slow_scan = time.time()
            st.toast("✅ Deep scan complete — all scanners refreshed", icon="🔄")
    except Exception as _deep_e:
        # Never let a stage failure stall the sequence — skip to next stage
        _order = ["sector", "universe", "smc", "traps", "vcp", "rs"]
        try:
            _nxt = _order[_order.index(_stage) + 1]
            st.session_state._deep_stage = _nxt
        except Exception:
            st.session_state._deep_running = False
            st.session_state._deep_stage = "sector"
            st.session_state.last_slow_scan = time.time()
        st.toast(f"⚠️ {_stage} stage error: {_deep_e}", icon="⚠️")

    # Chain the next stage (or finish) with an immediate rerun
    if st.session_state.get("_deep_running", False):
        st.rerun()

# ── First-paint kickoff ────────────────────────────────────────────────────────
# After the very first render post-login, immediately rerun once so the fast
# scan (signals + news) fires right away instead of waiting for the next
# user interaction or autorefresh tick.
if st.session_state.get("_kickoff_scan", False):
    st.session_state._kickoff_scan = False
    st.rerun()
