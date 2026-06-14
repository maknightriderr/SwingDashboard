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

# ── Performance: @st.cache_data wrappers ──────────────────────────────────────
# market_regime is global (same for all users) — safe to cache across sessions.
# TTL 600s = 10 min. This renders the header banner in <100ms on reruns.
@st.cache_data(ttl=600, show_spinner=False)
def _cached_market_regime():
    return get_market_regime()

# Price cache: 5-min TTL so KPI cards don't block on every sidebar interaction.
@st.cache_data(ttl=300, show_spinner=False)
def _cached_prices(symbols_tuple):
    """Fetch live prices for a tuple of symbols. Tuple is hashable → cacheable.
    Robust across yfinance versions: tries fast_info (dict OR attribute style),
    then history() as a fallback. Never raises — missing prices just stay absent."""
    import yfinance as _yf
    import pandas as _pd
    prices = {}

    def _extract_fast_price(t):
        """Get last price from fast_info regardless of yfinance API shape."""
        fi = None
        try:
            fi = t.fast_info
        except Exception:
            return None
        # Try several access styles — yfinance changed this API across versions
        for key in ("last_price", "lastPrice", "regularMarketPrice"):
            # dict-style .get
            try:
                v = fi.get(key) if hasattr(fi, "get") else None
                if v is not None and not _pd.isna(v):
                    return float(v)
            except Exception:
                pass
            # attribute-style
            try:
                v = getattr(fi, key, None)
                if v is not None and not _pd.isna(v):
                    return float(v)
            except Exception:
                pass
            # key-index style
            try:
                v = fi[key]
                if v is not None and not _pd.isna(v):
                    return float(v)
            except Exception:
                pass
        return None

    for sym in symbols_tuple:
        clean = str(sym).upper().strip()
        for sfx in [".NS", ".BO", ".NSE", ".BSE"]:
            if clean.endswith(sfx):
                clean = clean[:-len(sfx)]
        got = False
        for sfx in [".NS", ".BO"]:
            try:
                t = _yf.Ticker(clean + sfx)
                # 1) fast_info
                v = _extract_fast_price(t)
                if v is not None and v > 0:
                    prices[sym] = round(v, 2); got = True; break
                # 2) recent history (5d covers weekends/holidays)
                h = t.history(period="5d", interval="1d", auto_adjust=True)
                if h is not None and not h.empty and "Close" in h.columns:
                    last_valid = h["Close"].dropna()
                    if not last_valid.empty:
                        prices[sym] = round(float(last_valid.iloc[-1]), 2)
                        got = True; break
            except Exception:
                continue
        # leave missing if both suffixes failed
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
    return hashlib.sha256(str.encode(password + "swing_salt_99")).hexdigest()

def verify_hash(password, hashed_pw):
    return make_hash(password) == hashed_pw

# ── Database ───────────────────────────────────────────────────────────────────
# PERSISTENCE STRATEGY:
#   Streamlit Cloud has an EPHEMERAL filesystem — a local SQLite .db file is
#   wiped on every restart/redeploy/sleep, which flushes all your trades & logins.
#   Fix: use a hosted Postgres DB when a connection string is provided in
#   st.secrets (key: DATABASE_URL or [postgres].url), which PERSISTS across
#   restarts. Falls back to local SQLite when no Postgres is configured (so it
#   still runs locally / in dev).
#
#   To make your data persist on Streamlit Cloud, add to your app secrets:
#       DATABASE_URL = "postgresql://user:pass@host:5432/dbname"
#   (Free Postgres: Supabase or Neon. Copy their connection string.)
# ==============================================================================

DB = "trades_v2.db"   # SQLite fallback path

def _get_pg_url():
    """Return a Postgres connection URL from secrets, or None for SQLite mode."""
    try:
        if "DATABASE_URL" in st.secrets:
            return st.secrets["DATABASE_URL"]
        if "postgres" in st.secrets and "url" in st.secrets["postgres"]:
            return st.secrets["postgres"]["url"]
    except Exception:
        pass
    return None

_PG_URL = _get_pg_url()
_USE_PG = _PG_URL is not None

# Lazy import psycopg2 only if Postgres is configured
if _USE_PG:
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        _USE_PG = False   # psycopg2 not installed → fall back to SQLite

def _pg_conn():
    """Open a Postgres connection."""
    return psycopg2.connect(_PG_URL, sslmode="require", connect_timeout=10)


def _q(sql):
    """Translate SQLite '?' placeholders to Postgres '%s' when in PG mode."""
    if _USE_PG:
        return sql.replace("?", "%s")
    return sql

def init_db():
    global _USE_PG
    if _USE_PG:
        try:
            conn = _pg_conn(); cur = conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS users(
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL)""")
            cur.execute("""CREATE TABLE IF NOT EXISTS trades(
                id SERIAL PRIMARY KEY, user_id INTEGER, stock TEXT NOT NULL,
                quantity REAL NOT NULL, buy_at REAL NOT NULL, sell_at REAL,
                status TEXT DEFAULT 'Open',
                added_date TEXT DEFAULT to_char(CURRENT_DATE,'YYYY-MM-DD'),
                closed_date TEXT)""")
            cur.execute("""CREATE TABLE IF NOT EXISTS portfolio_history(
                id SERIAL PRIMARY KEY, user_id INTEGER, snapshot_date TEXT,
                total_invested REAL, current_value REAL)""")
            cur.execute("""CREATE TABLE IF NOT EXISTS tg_config(
                user_id INTEGER PRIMARY KEY, bot_token TEXT, chat_id TEXT)""")
            cur.execute("""CREATE TABLE IF NOT EXISTS watchlist(
                id SERIAL PRIMARY KEY, user_id INTEGER, stock TEXT NOT NULL,
                target_price REAL, notes TEXT,
                added_date TEXT DEFAULT to_char(CURRENT_DATE,'YYYY-MM-DD'))""")
            conn.commit(); cur.close(); conn.close()
            return
        except Exception as _e:
            _USE_PG = False
            try:
                st.warning(
                    "⚠️ Postgres unavailable — falling back to local SQLite "
                    "(data will NOT persist across restarts on Streamlit Cloud). "
                    "If using Supabase, use the **Session Pooler** connection "
                    "string (IPv4-compatible), not the direct one. "
                    f"Details: {_e}"
                )
            except Exception:
                pass
    # SQLite path (default, and fallback if Postgres failed above)
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
    c.commit(); c.close()



def db(sql, params=(), fetch=False):
    if _USE_PG:
        conn = _pg_conn(); cur = conn.cursor()
        cur.execute(_q(sql), params)
        conn.commit()
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

def get_tg_config(user_id):
    rows = db("SELECT bot_token,chat_id FROM tg_config WHERE user_id=?",
              (user_id,), fetch=True)
    return rows[0] if rows else ("", "")

def save_tg_config(user_id, token, chat):
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
    today = datetime.now().strftime("%Y-%m-%d")
    db("DELETE FROM portfolio_history WHERE snapshot_date=? AND user_id=?", (today, user_id))
    db("INSERT INTO portfolio_history(user_id,snapshot_date,total_invested,current_value) VALUES(?,?,?,?)",
       (user_id, today, invested, value))

def add_watchlist(user_id, stock, target=None, notes=""):
    db("INSERT INTO watchlist(user_id,stock,target_price,notes) VALUES(?,?,?,?)",
       (user_id, stock.upper().strip(), target, notes))

def get_watchlist(user_id):
    if _USE_PG:
        conn = _pg_conn()
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

def delete_watchlist_item(wid, user_id):
    db("DELETE FROM watchlist WHERE id=? AND user_id=?", (wid, user_id))

# ── Session & Cookie Init ──────────────────────────────────────────────────────
from streamlit_cookies_controller import CookieController
controller = CookieController(key='app_cookies')
init_db()

# Ensure session state variables exist
for k, v in [("user_id", None), ("username", None), ("edit_id", None), ("close_id", None), ("del_id", None),
             ("last_refresh", None), ("last_auto_scan", 0.0), ("last_slow_scan", 0.0),
             ("_trade_hash", -1), ("sort_col", "stock"), ("sort_asc", False),
             ("signals_cache", None), ("sector_cache", None), ("picks_cache", None),
             ("outlook_cache", None), ("scanner_cache", None), ("trap_scan_cache", None),
             ("corp_actions_cache", None), ("selected_scanner_sector", "All Sectors"),
             ("custom_stocks_input", ""), ("active_page", "portfolio"),
              ("smc_scan_cache", None),
             ("first_render_done", False), ("_kickoff_scan", False),
             ("_scan_stage", "done"),
             ("_deep_stage", "idle"),
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
            cookie_uid = int(cookies.get("swing_user_id"))
            user_row = db("SELECT username FROM users WHERE id=?",
                          (cookie_uid,), fetch=True)
            if user_row:
                st.session_state.user_id = cookie_uid
                st.session_state.username = user_row[0][0]
                st.session_state.first_render_done = False  # defer scans
                st.rerun()
            else:
                # Cookie points to a user that doesn't exist in THIS database
                # (e.g. Postgres→SQLite fallback). Clear the stale cookie.
                controller.set("swing_user_id", "", max_age=0)
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
                        controller.set("swing_user_id", str(uid), max_age=604800)
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
}

# --- Fail-safe to prevent KeyErrors when a saved theme name no longer exists ---
if st.session_state.theme not in THEMES:
    st.session_state.theme = "Obsidian & Gold (Institutional)"

def theme_css(t):
    glow  = t.get("glow", "rgba(255,255,255,0.1)")
    bg_fx = t.get("bg_fx", "none")
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

:root {{
  --bg:{t['bg']}; --card:{t['card']}; --input:{t['input']};
  --border:{t['border']}; --text:{t['text']}; --muted:{t['muted']};
  --green:{t['green']}; --red:{t['red']}; --yellow:{t['yellow']};
  --blue:{t['blue']}; --accent:{t['accent']}; --card2:{t['card2']};
  --gradient:{t['gradient']}; --glow:{glow};
}}

/* ═══ Base canvas with ambient light bloom ═══ */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
    background: var(--bg) !important; color: var(--text) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    -webkit-font-smoothing: antialiased;
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
    font-family: 'Plus Jakarta Sans', sans-serif !important; font-weight: 600 !important;
}}
button[data-testid="stNumberInputStepDown"], button[data-testid="stNumberInputStepUp"] {{
    background-color: var(--card2) !important; color: var(--text) !important; border: none !important;
}}
div[role="listbox"] {{ background-color: var(--card2) !important;
    border: 1px solid var(--border) !important; border-radius: 10px !important; }}
ul[role="listbox"] li {{ color: var(--text) !important; font-weight: 500 !important; }}
ul[role="listbox"] li[aria-selected="true"] {{
    background-color: var(--accent) !important; color: #000 !important; font-weight: 800 !important; }}

.stButton>button {{
    background: var(--card2) !important; border: 1px solid var(--border) !important;
    color: var(--text) !important; border-radius: 10px !important; font-weight: 700 !important;
    letter-spacing: .05em !important; padding: .6rem 1.2rem !important;
    transition: all .3s ease !important;
}}
.stButton>button:hover {{
    border-color: var(--accent) !important; background: var(--accent) !important;
    color: #000 !important; box-shadow: 0 0 24px var(--glow) !important;
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
            h = t.history(period="5d", interval="1d", auto_adjust=True)
            if h is not None and not h.empty and "Close" in h.columns:
                lv = h["Close"].dropna()
                if not lv.empty:
                    p = round(float(lv.iloc[-1]), 2)
                    _CACHE[clean] = (p, time.time())
                    return p
        except Exception:
            continue
    return None


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
def base_layout(fig, title):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#f8fafc", weight="bold"), x=.01),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", size=11), margin=dict(l=8, r=8, t=45, b=8))
    fig.update_xaxes(gridcolor="#334155", zerolinecolor="#334155", tickfont=dict(color="#cbd5e1"))
    fig.update_yaxes(gridcolor="#334155", zerolinecolor="#334155", tickfont=dict(color="#cbd5e1"))
    return fig


def chart_alloc(df):
    g = df.groupby("stock")["invested"].sum().reset_index()
    return base_layout(go.Figure(go.Pie(
        labels=g["stock"], values=g["invested"], hole=0.4,
        marker=dict(colors=px.colors.qualitative.Dark24,
                    line=dict(color="rgba(0,0,0,0)", width=0)),
        textinfo="percent+label", textfont=dict(size=11, color="#ffffff")
    )), "Portfolio Allocation")


def chart_pnl(df):
    d = df.sort_values("profit")
    fig = base_layout(go.Figure(go.Bar(
        x=d["profit"], y=d["stock"], orientation="h",
        marker=dict(color=["#ef4444" if v < 0 else "#10b981" for v in d["profit"]],
                    line=dict(width=0)),
        text=[fp(p) for p in d["profit_pct"]],
        textposition="outside", textfont=dict(color="#f8fafc", size=10)
    )), "P&L by Stock")
    fig.update_layout(showlegend=False, margin=dict(l=8, r=55, t=45, b=8))
    fig.update_xaxes(tickprefix="₹")
    return fig


def chart_donut(df):
    c = df["status"].value_counts().reset_index()
    c.columns = ["Status", "Count"]
    fig = base_layout(go.Figure(go.Pie(
        labels=c["Status"], values=c["Count"], hole=.6,
        marker=dict(
            colors=[{"Open": "#f59e0b", "Closed": "#10b981"}.get(s, "#94a3b8")
                    for s in c["Status"]],
            line=dict(color="rgba(0,0,0,0)", width=0)),
        textinfo="percent+value", textfont=dict(size=12, color="#ffffff")
    )), "Open vs Closed")
    fig.add_annotation(
        text=f"<b>{len(df)}</b><br><span style='font-size:10px'>TRADES</span>",
        font=dict(size=18, color="#f8fafc"), showarrow=False, x=.5, y=.5)
    return fig


def chart_growth(hist, cur_val, cur_inv):
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
                   name="Value", line=dict(color="#10b981", width=3),
                   fill="tozeroy", fillcolor="rgba(16,185,129,0.1)"),
        go.Scatter(x=pd.to_datetime(d["snapshot_date"]), y=d["total_invested"],
                   name="Invested", line=dict(color="#3b82f6", width=2, dash="dash"))
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

        html += f"""
<div class="sig-card {c}">
  <div class="sig-action" style="color:{clr}">{action}</div>
  <div style="font-size:.9rem;font-weight:800;margin-bottom:.3rem">
    {s.get('stock','')}
    <span style="font-size:.7rem;color:var(--muted);font-weight:400">
      {s.get('sector','')}
    </span>
  </div>
  <div class="sig-meta">CMP {cmp_str} · RSI {rsi_str} · {pct_str}</div>
  <div class="sig-reason">{reason}</div>
  <div class="sig-price">{ph}</div>
  <div class="str-bar">
    <div class="str-fill" style="width:{strength}%;background:{clr}"></div>
  </div>
</div>"""

    st.markdown(html + "</div>", unsafe_allow_html=True)


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
    ]
    avg = sum(s[1] for s in scores) / len(scores)

    st.markdown(f"""
    <div style="background:var(--card);border:1px solid var(--border);border-radius:12px;
         padding:1.2rem 1.5rem;margin-bottom:1.5rem">
      <div style="font-size:.75rem;color:var(--muted);text-transform:uppercase;
           letter-spacing:.08em;margin-bottom:.5rem">signals.py v12 — Overall Score</div>
      <div style="font-size:2.5rem;font-weight:800;color:var(--accent)">{avg:.1f}<span
           style="font-size:1rem;color:var(--muted);font-weight:400"> / 10</span></div>
      <div style="font-size:.8rem;color:var(--muted);margin-top:.3rem">
        v11 (6.6) → v12 (8.4). Every component now ≥ 8: Wilder ATR/RSI, numpy
        Supertrend, 20-day VWAP, swing-peak Fibonacci, unified risk engine.
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

if not st.session_state.get("first_render_done", False):
    # PASS 1 — first paint after login: render immediately, defer ALL scanning.
    st.session_state.first_render_done = True
    _fast_due = False
    _deep_due = False
    st.session_state._kickoff_scan = True
    st.session_state._scan_stage = "fast"   # next pass does the fast scan
elif st.session_state.get("_scan_stage") == "fast":
    # PASS 2 — fast scan only (signals + news, ~20-40s). Deep scan still deferred
    # so the core dashboard becomes usable before the heavy universe sweep.
    _fast_due = True
    _deep_due = False
    st.session_state._scan_stage = "deep"
    st.session_state._kickoff_scan = True   # one more rerun to start deep scan
elif st.session_state.get("_scan_stage") == "deep":
    # PASS 3 — deep scan (sector + universe + SMC). After this, normal timers.
    _fast_due = False
    _deep_due = True
    st.session_state._scan_stage = "done"
else:
    # Steady state — scans fire only on their 5-min / 15-min schedules.
    _fast_due = (st.session_state.last_auto_scan == 0.0 or
                 (_now - st.session_state.last_auto_scan) >= 300)   # 5 min
    _deep_due = (st.session_state.last_slow_scan == 0.0 or
                 (_now - st.session_state.last_slow_scan) >= 900)    # 15 min

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

# ── Staged deep scan: sector → universe → SMC, one stage per rerun ──────────
# When the 15-min timer fires (PASS 3 or steady-state sets _deep_due), we start
# the pipeline at the "sector" stage. Each stage runs alone, paints, then
# triggers a quick rerun to advance — so the UI never blocks on all three.
if _deep_due and st.session_state._deep_stage == "idle":
    st.session_state._deep_stage = "sector"

_stage = st.session_state._deep_stage

if _stage == "sector":
    with st.spinner("🔄 Deep scan 1/3: sector rotation…"):
        try:
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
        except Exception as _e:
            st.toast(f"⚠️ Sector refresh error: {_e}", icon="⚠️")
    st.session_state._deep_stage = "universe"
    st.session_state._kickoff_scan = True

elif _stage == "universe":
    with st.spinner("🔄 Deep scan 2/3: universe scanner…"):
        try:
            _usd = generate_market_scanner()
            st.session_state.scanner_cache = (_usd if (_usd is not None and not _usd.empty)
                                              else pd.DataFrame())
        except Exception as _e:
            st.toast(f"⚠️ Universe scan error: {_e}", icon="⚠️")
    st.session_state._deep_stage = "smc"
    st.session_state._kickoff_scan = True

elif _stage == "smc":
    with st.spinner("🔄 Deep scan 3/3: SMC setups…"):
        try:
            if scan_for_smc_setups is not None:
                st.session_state.smc_scan_cache = scan_for_smc_setups(
                    min_quality="B", action_filter="All")
        except Exception as _e:
            st.toast(f"⚠️ SMC scan error: {_e}", icon="⚠️")
    st.session_state._deep_stage = "idle"          # cycle complete
    st.session_state.last_slow_scan = _now         # restart 15-min timer

# ── Portfolio metrics ──────────────────────────────────────────────────────────
if not df.empty:
    odf = df[df["status"] == "Open"]
    cdf = df[df["status"] == "Closed"]
    t_inv    = df["invested"].sum()
    t_cur    = df["current_amt"].sum()
    t_real   = cdf["profit"].sum()   if not cdf.empty else 0
    t_unreal = odf["profit"].sum()   if not odf.empty else 0
    t_pnl    = df["profit"].sum()
    t_pnl_pct = t_pnl / t_inv * 100 if t_inv > 0 else 0
    best  = df.loc[df["profit_pct"].idxmax(), "stock"]
    worst = df.loc[df["profit_pct"].idxmin(), "stock"]
    save_snapshot(UID, t_inv, t_cur)
else:
    odf = cdf = pd.DataFrame()
    t_inv = t_cur = t_real = t_unreal = t_pnl = t_pnl_pct = 0
    best = worst = "—"

theme_t = THEMES[st.session_state.theme]
st.markdown(theme_css(theme_t), unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div style="font-size:.85rem;font-weight:800;color:var(--accent);'
        f'margin-bottom:1rem">👤 {(st.session_state.username or "USER").upper()}</div>',

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
        st.rerun()

    st.markdown("<hr style='margin:.8rem 0;border-color:var(--border)'>",
                unsafe_allow_html=True)
    st.markdown('<div style="font-size:.8rem;font-weight:800;letter-spacing:.05em;'
                'margin-bottom:.5rem">🗺 NAVIGATION</div>', unsafe_allow_html=True)

    NAV_GROUPS = {
        "📊 Portfolio": [
            ("📋 Overview",          "portfolio"),
            ("📊 Charts",            "analytics"),
            ("📐 Metrics",           "metrics"),
            ("📤 Export",            "export"),
        ],
        "🔔 Signals & Alerts": [
            ("🔔 Active Signals",    "signals"),
            ("🪤 Trap Scanner",      "traps"),
            ("🏦 Smart Money (SMC)", "smc"),
        ],
        "🔄 Market Intelligence": [
            ("🔄 Sector Rotation",   "sector"),
            ("🌌 Universe Scanner",  "scanner"),
            ("📅 Corporate Actions", "corp_actions"),
        ],
        "🛠 Tools": [
            ("👁 Watchlist",         "watchlist"),
            ("🎯 Signal Scores",     "scores"),
        ],
    }
    # Flat list for radio
    nav_labels = [label for group in NAV_GROUPS.values() for label, _ in group]
    nav_keys   = [key   for group in NAV_GROUPS.values() for _, key   in group]

    if "active_page" not in st.session_state:
        st.session_state.active_page = "portfolio"

    # Group headers + radio buttons styled with CSS
    nav_html = ""
    flat_idx = 0
    for group_label, items in NAV_GROUPS.items():
        nav_html += (f'<div style="font-size:.65rem;font-weight:800;color:var(--muted);'
                     f'text-transform:uppercase;letter-spacing:.1em;margin:.6rem 0 .2rem;'
                     f'padding-left:.3rem">{group_label}</div>')
        flat_idx += len(items)

    # Use radio for actual selection (CSS handles grouping visually)
    cur_idx = nav_keys.index(st.session_state.active_page) \
              if st.session_state.active_page in nav_keys else 0
    sel_nav = st.radio(
        "nav", nav_labels, index=cur_idx,
        label_visibility="collapsed")
    new_page = nav_keys[nav_labels.index(sel_nav)]
    if new_page != st.session_state.active_page:
        st.session_state.active_page = new_page
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
                "💾 Update Trade" if em else "➕ Execute Entry", width="stretch"):
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
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Force Scan", width="stretch"):
            _cached_market_regime.clear()
            _cached_prices.clear()
            st.session_state.last_auto_scan = 0.0
            st.session_state.last_slow_scan = 0.0
            st.session_state._trade_hash    = -1
            st.rerun()
    with c2:
        _elapsed_fast = time.time() - st.session_state.last_auto_scan
        _elapsed_slow = time.time() - st.session_state.last_slow_scan
        _nxt_fast = max(0, int((300 - _elapsed_fast) // 60))
        _nxt_slow = max(0, int((900 - _elapsed_slow) // 60))
        st.markdown(
            f'<div style="font-size:.7rem;color:var(--muted);padding-top:.5rem;'
            f'font-weight:600;line-height:1.6">'
            f'⚡ {_nxt_fast}m core · 🔄 {_nxt_slow}m deep</div>',
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
    '<div class="dash-title-text">📈 Quantitative <span class="hl">Swing Dashboard</span></div>'
    '<span class="refresh-badge">⚡ SIGNALS LIVE · 🔄 SECTOR LIVE</span>'
    '</div>',
    unsafe_allow_html=True)

market = _cached_market_regime()
regime = market["regime"]

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
for name, d in market.get("indices", {}).items():
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
pnl_c = "green" if t_pnl >= 0 else "red"
r_c   = "green" if t_real >= 0 else "red"
u_c   = "green" if t_unreal >= 0 else "red"

st.markdown(
    '<div class="cards">'
    + card("Total Invested",  fi(t_inv),    "",          "blue")
    + card("Portfolio Value", fi(t_cur),    "",          "blue")
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

        render_signals(st.session_state.signals_cache, theme_t)

# ── Sector Rotation ──────────────────────────────────────────────────────────
elif _page == 'sector':
    st.markdown('<div class="sec">Macro Sector Rotation & Capital Flow</div>',
                unsafe_allow_html=True)

    if st.session_state.sector_cache is not None:
        render_sector(st.session_state.sector_cache, theme_t)

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
            st.rerun()

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
                "Patterns":st.column_config.TextColumn("Patterns",width="large"),
            })

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
    st.markdown('<div class="sec">🎯 signals.py v12 — Component Scorecard</div>',
                unsafe_allow_html=True)
    render_score_dashboard()
    st.markdown("""
<div style="margin-top:1rem;padding:1rem;background:rgba(16,185,129,.08);
     border:1px solid rgba(16,185,129,.3);border-radius:8px;font-size:.85rem;
     color:var(--muted);line-height:1.8">
<b style="color:var(--text)">✅ All v12 priority fixes shipped:</b><br>
1. <b>MACD</b> — single-pass crossover + histogram momentum flags<br>
2. <b>Supertrend</b> — numpy array loop, Wilder ATR(10), mult 2.5<br>
3. <b>ATR</b> — Wilder's EWM smoothing (matches Zerodha/TradingView)<br>
4. <b>RSI</b> — adjust=False on all ewm() + explicit 100/0 edges<br>
5. <b>VWAP</b> — 20-day rolling + price_vs_vwap deviation<br>
6. <b>Fibonacci</b> — scipy swing-peak detection, not fixed window<br>
7. <b>Risk Engine</b> — unified across signals, picks, and scanner
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

# ── Post-render scan kickoff ───────────────────────────────────────────────────
# The dashboard has now fully rendered. If this was the first paint after login,
# trigger ONE immediate rerun so the deferred scans begin on the next pass —
# the user sees a complete dashboard instantly, then data fills in moments later.
if st.session_state.get("_kickoff_scan", False):
    st.session_state._kickoff_scan = False
    time.sleep(0.1)
    st.rerun()
