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
    cookies = controller.getAll()

    if cookies is None:
        st.info("Loading secure tunnel...")
        time.sleep(0.5)
        st.rerun()

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
    t_inv    = odf["invested"].sum()    