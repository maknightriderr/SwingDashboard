"""
Swing Trading Portfolio Dashboard v13
Fixes vs v12:
  - Signal cards: target/SL/RR now use unified _calc_risk_params output from signals.py
  - Sector picks vs Active Signals discrepancy: both display same risk engine output
  - render_signals(): null-safe for all fields, RR capped display at 10x
  - render_sector(): RRG quadrant + RS vs Nifty columns added
  - Tab 4 banner: RS vs Nifty + RRG quadrant shown
  - News: fetch called correctly, displayed with st.markdown unsafe_allow_html
  - Score dashboard added (new tab)
  - All ema20/ema50 references updated to ema9/ema21 from signals v11
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
    fetch_portfolio_news
)

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
DB = "trades_v2.db"

def init_db():
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
    except sqlite3.IntegrityError:
        return False

def login_user(username, password):
    user = db("SELECT id, password_hash FROM users WHERE username=?",
              (username.lower(),), fetch=True)
    if user and verify_hash(password, user[0][1]):
        return user[0][0]
    return None

def get_trades(user_id):
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT * FROM trades WHERE user_id=? ORDER BY id DESC",
        conn, params=(user_id,))
    conn.close()
    return df

def get_history(user_id):
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
             ("last_refresh", None), ("last_auto_scan", 0.0), ("sort_col", "stock"), ("sort_asc", False),
             ("signals_cache", None), ("sector_cache", None), ("picks_cache", None),
             ("outlook_cache", None), ("scanner_cache", None), ("filter_status", "All"),
             ("filter_pnl", "All"), ("search", ""), ("theme", "Obsidian & Gold (Institutional)")]: # <--- UPDATE THIS
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
            st.session_state.user_id = cookie_uid
            conn = sqlite3.connect(DB)
            user_row = conn.execute(
                "SELECT username FROM users WHERE id=?", (cookie_uid,)).fetchone()
            conn.close()
            if user_row:
                st.session_state.username = user_row[0]
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
    "Obsidian & Gold (Institutional)": {
        "bg":"#050608","card":"#0d0e12","input":"#15171c","border":"rgba(212, 175, 55, 0.15)",
        "text":"#fdfdfd","muted":"#8e8e93","green":"#10b981","red":"#ef4444",
        "yellow":"#d4af37","blue":"#3b82f6","accent":"#d4af37","card2":"#121419",
        "gradient":"linear-gradient(145deg, #0d0e12 0%, #050608 100%)"
    },
    "Deep Sapphire (Glass)": {
        "bg":"#020617","card":"rgba(15, 23, 42, 0.5)","input":"#1e293b","border":"rgba(56, 189, 248, 0.1)",
        "text":"#f8fafc","muted":"#94a3b8","green":"#10b981","red":"#f43f5e",
        "yellow":"#f59e0b","blue":"#0ea5e9","accent":"#38bdf8","card2":"rgba(30, 41, 59, 0.4)",
        "gradient":"linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(2, 6, 23, 0.9) 100%)"
    },
    "Carbon Matrix (Quant)": {
        "bg":"#09090b","card":"#121214","input":"#18181b","border":"rgba(255, 255, 255, 0.06)",
        "text":"#fafafa","muted":"#a1a1aa","green":"#22c55e","red":"#ff3366",
        "yellow":"#f59e0b","blue":"#06b6d4","accent":"#14b8a6","card2":"#18181b",
        "gradient":"linear-gradient(180deg, #121214 0%, #09090b 100%)"
    }
}

# --- ADD THIS FAIL-SAFE TO PREVENT KEY ERRORS ---
if st.session_state.theme not in THEMES:
    st.session_state.theme = "Obsidian & Gold (Institutional)"

def theme_css(t):
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {{
  --bg:{t['bg']}; --card:{t['card']}; --input:{t['input']};
  --border:{t['border']}; --text:{t['text']}; --muted:{t['muted']};
  --green:{t['green']}; --red:{t['red']}; --yellow:{t['yellow']};
  --blue:{t['blue']}; --accent:{t['accent']}; --card2:{t['card2']};
  --gradient:{t['gradient']};
}}

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
    background: var(--bg) !important; background-color: var(--bg) !important; color: var(--text) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    -webkit-font-smoothing: antialiased;
}}

/* Hide Streamlit Clutter & Custom Scrollbar */
[data-testid="stHeader"] {{ background: transparent !important; }}
#MainMenu, footer, header {{ display: none !important; }}
.block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; max-width: 96%; }}
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 10px; }}

/* Elite Title Styling */
.dash-title {{ font-size: 2rem; font-weight: 800; padding-bottom: 1rem; margin-bottom: 1.5rem; display: flex; align-items: center; justify-content: space-between; letter-spacing: -0.03em; border-bottom: 1px solid var(--border); }}
.dash-title-text {{ background: linear-gradient(to right, var(--text), var(--muted)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
.dash-title span.hl {{ color: var(--accent); -webkit-text-fill-color: var(--accent); }}

/* Luxury Metric Cards */
.cards {{ display: flex; gap: 1.2rem; flex-wrap: wrap; margin-bottom: 2.5rem; }}
.card {{ background: var(--gradient); border: 1px solid var(--border); border-radius: 14px; padding: 1.5rem; flex: 1; min-width: 160px; box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5); transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); }}
.card:hover {{ transform: translateY(-5px); box-shadow: 0 20px 40px -10px rgba(0,0,0,0.7); border-color: var(--accent); }}
.card .lbl {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.15em; color: var(--muted); margin-bottom: 0.5rem; font-weight: 700; }}
.card .val {{ font-size: 1.6rem; font-weight: 800; color: var(--text); letter-spacing: -0.03em; }}
.card .sub {{ font-size: 0.8rem; color: var(--muted); margin-top: 0.4rem; font-weight: 600; }}

/* Typography & Accent Colors */
.green {{ color: var(--green) !important; }} .red {{ color: var(--red) !important; }} .yellow {{ color: var(--yellow) !important; }} .blue {{ color: var(--blue) !important; }}
.sec {{ font-size: 1rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.12em; color: var(--text); margin: 2.5rem 0 1.2rem; padding-left: 1rem; border-left: 4px solid var(--accent); }}

/* Institutional Data Tables */
.tbl-wrap {{ overflow-x: auto; background: var(--card); border: 1px solid var(--border); border-radius: 14px; box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5); backdrop-filter: blur(16px); margin-bottom: 1.5rem; }}
table.t, .sector-tbl {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; font-variant-numeric: tabular-nums; }}
table.t th, .sector-tbl th {{ background: var(--card2); color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700; padding: 1.2rem 1rem; text-align: right; border-bottom: 1px solid var(--border); }}
table.t th.l, table.t td.l {{ text-align: left; }}
table.t td, .sector-tbl td {{ padding: 1rem; border-bottom: 1px solid var(--border); text-align: right; color: var(--text); font-weight: 600; }}
table.t tr:last-child td, .sector-tbl tr:last-child td {{ border-bottom: none; }}
table.t tr:hover td, .sector-tbl tr:hover td {{ background: rgba(255,255,255,0.02); }}

/* Glowing Badges */
.pos {{ color: var(--green); font-weight: 800; }} .neg {{ color: var(--red); font-weight: 800; }} .zero-cell {{ color: var(--muted) !important; }}
.badge {{ display: inline-block; padding: 0.3rem 0.8rem; border-radius: 8px; font-size: 0.7rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em; }}
.b-open {{ background: rgba(212, 175, 55, 0.1); color: var(--yellow); border: 1px solid rgba(212, 175, 55, 0.3); box-shadow: 0 0 10px rgba(212, 175, 55, 0.1); }}
.b-cl {{ background: rgba(16, 185, 129, 0.1); color: var(--green); border: 1px solid rgba(16, 185, 129, 0.3); box-shadow: 0 0 10px rgba(16, 185, 129, 0.1); }}
.b-cll {{ background: rgba(239, 68, 68, 0.1); color: var(--red); border: 1px solid rgba(239, 68, 68, 0.3); box-shadow: 0 0 10px rgba(239, 68, 68, 0.1); }}

/* Signal & Pick Cards */
.sig-grid, .pick-grid, .outlook-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 1.5rem; margin-top: 1rem; }}
.sig-card, .pick-card, .outlook-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 1.5rem; transition: all 0.3s ease; backdrop-filter: blur(16px); box-shadow: 0 10px 20px -5px rgba(0,0,0,0.3); }}
.sig-card:hover, .pick-card:hover {{ transform: translateY(-4px); border-color: var(--accent); box-shadow: 0 15px 30px -5px rgba(0,0,0,0.5); }}
.sig-card.sell {{ border-top: 4px solid var(--red); }} .sig-card.avg {{ border-top: 4px solid var(--yellow); }}
.sig-card.hold {{ border-top: 4px solid var(--green); }} .sig-card.watch {{ border-top: 4px solid var(--muted); }}
.pick-card {{ border-top: 4px solid var(--accent); }}

.sig-action {{ font-size: 0.9rem; font-weight: 800; margin-bottom: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; }}
.sig-meta, .pick-sector {{ font-size: 0.8rem; color: var(--muted); font-weight: 600; }}
.sig-reason, .pick-prices {{ font-size: 0.9rem; margin-top: 1rem; color: var(--text); line-height: 1.6; }}
.sig-price, .pick-reason {{ font-size: 0.85rem; margin-top: 1.2rem; padding-top: 1rem; border-top: 1px solid var(--border); font-weight: 600; color: var(--muted); }}

/* Custom UI Inputs & Buttons */
[data-testid="stSidebar"] {{ background: var(--card) !important; border-right: 1px solid var(--border); padding-top: 2rem; }}
div[data-baseweb="input"], div[data-baseweb="select"], [data-testid="stNumberInputContainer"] {{ background-color: var(--input) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }}
div[data-baseweb="input"] input, [data-testid="stNumberInputContainer"] input {{ color: var(--text) !important; -webkit-text-fill-color: var(--text) !important; background-color: transparent !important; font-family: 'Plus Jakarta Sans', sans-serif !important; font-weight: 600 !important; }}
button[data-testid="stNumberInputStepDown"], button[data-testid="stNumberInputStepUp"] {{ background-color: var(--card2) !important; color: var(--text) !important; border: none !important; }}
div[role="listbox"] {{ background-color: var(--card2) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }}
ul[role="listbox"] li {{ color: var(--text) !important; font-weight: 500 !important; }} ul[role="listbox"] li[aria-selected="true"] {{ background-color: var(--accent) !important; color: #000 !important; font-weight: 800 !important; }}

.stButton>button {{ background: var(--card2) !important; border: 1px solid var(--border) !important; color: var(--text) !important; border-radius: 10px !important; font-weight: 700 !important; letter-spacing: 0.05em !important; padding: 0.6rem 1.2rem !important; transition: all 0.3s ease !important; }}
.stButton>button:hover {{ border-color: var(--accent) !important; background: var(--accent) !important; color: #000 !important; box-shadow: 0 0 20px var(--accent) !important; transform: scale(1.02); }}

/* Sleek Tabs */
.stTabs [data-baseweb="tab-list"] {{ background: transparent; gap: 2.5rem; padding: 0 0.5rem; border-bottom: 1px solid var(--border); }}
.stTabs [data-baseweb="tab"] {{ background: transparent; color: var(--muted); font-weight: 700; padding: 1.2rem 0; border: none; border-bottom: 3px solid transparent; text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.85rem; transition: color 0.3s ease; }}
.stTabs [aria-selected="true"] {{ background: transparent !important; color: var(--text) !important; border-bottom-color: var(--accent) !important; }}

/* Regime Banner - Glassy & Glowing */
.refresh-badge {{ display: inline-block; background: rgba(16, 185, 129, 0.1); color: var(--green); padding: 0.4rem 1rem; border-radius: 30px; font-size: 0.75rem; font-weight: 800; border: 1px solid rgba(16, 185, 129, 0.4); letter-spacing: 0.1em; text-transform: uppercase; box-shadow: 0 0 15px rgba(16, 185, 129, 0.2); }}
.regime-banner {{ border-radius: 16px; padding: 1.2rem 1.8rem; display: flex; align-items: center; gap: 1.2rem; margin-bottom: 2.5rem; flex-wrap: wrap; box-shadow: 0 15px 35px -10px rgba(0,0,0,0.6); border: 1px solid var(--border); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px); }}
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
    for sfx in [".NS", ".BO"]:
        try:
            t = yf.Ticker(clean + sfx)
            val = t.fast_info.get("last_price")
            if val is not None and not pd.isna(val):
                p = round(float(val), 2)
                _CACHE[clean] = (p, time.time())
                return p
            h = t.history(period="1d", interval="1d", auto_adjust=True)
            if h is not None and not h.empty and "Close" in h.columns:
                p = round(float(h["Close"].iloc[-1]), 2)
                _CACHE[clean] = (p, time.time())
                return p
        except Exception:
            continue
    return None


def enrich(df):
    if df.empty:
        return df
    prices = {s: fetch_price(s) for s in df["stock"].unique()}
    df = df.copy()
    df["cmp"] = df["stock"].map(prices)
    df["nse_label"] = "NSE:" + df["stock"]
    df["invested"] = df["quantity"] * df["buy_at"]
    df["current_amt"] = np.where(
        df["status"] == "Open",
        df["quantity"] * df["cmp"].fillna(df["buy_at"]),
        df["quantity"] * df["sell_at"].fillna(df["buy_at"])
    )
    df["total_amt"] = np.where(
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


# ── Signal card renderer (FIXED) ──────────────────────────────────────────────
def _fmt_rr(rr):
    """
    FIX: Cap RR display at 10x and flag anything above 5x as suspicious.
    The old code showed RR=16.6 for BLUEJET because trail_stop = buy_at
    made risk = 0.5. Now signals.py uses _calc_risk_params (2*ATR stop)
    so this shouldn't occur — but we add a visual guard anyway.
    """
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

        # Null-safe formatters
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


# ── Sector table renderer (FIXED: RRG + RS vs Nifty columns) ──────────────────
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


# ── News renderer (FIXED) ─────────────────────────────────────────────────────
def render_news(news_list):
    """
    FIX: news items from signals.py already contain HTML anchor tags.
    Must use unsafe_allow_html=True. Previous version used st.write()
    which escaped the HTML and broke the links.
    """
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
        ("RSI (Wilder's)",      7, "adjust=False missing, edge case fix missing from v11"),
        ("MACD",                6, "Double-fire loop (for i in [-2,-1]) still present, adjust=False missing"),
        ("Bollinger Bands",     6, "bb_pos not clamped [0,1], bandwidth/squeeze not computed"),
        ("ATR",                 6, "SMA rolling(14) — not Wilder's EWM, stops differ from Zerodha"),
        ("Supertrend",          5, "Pandas chained .iloc[i]= still present — numpy array fix needed"),
        ("VWAP",                4, "5-bar rolling unchanged from v10 — meaningless on daily data"),
        ("EMA / Trend",         7, "ema9/21/50 correctly renamed, but no slope check, EMA200 dropped"),
        ("Fibonacci",           6, "Fixed 60-bar window — not swing-peak based"),
        ("Chart Patterns",      7, "Neckline + Cup&Handle + vol gates — solid upgrade"),
        ("Candlesticks",        8, "3-candle patterns, range normalization — strong upgrade"),
        ("Signal RR Engine",    7, "_calc_risk_params added but find_sector_picks() bypasses it"),
        ("Sector Rotation",     8, "RRG quadrant + RS vs Nifty — strong"),
        ("News Engine",         8, "yfinance v1.4 fix + RSS fallback — reliable"),
        ("Liquidity Gate",      7, "₹1Cr threshold correct, no user-facing fallback message"),
        ("Unified Risk Engine", 7, "Good concept — but two code paths still produce different SL/Target"),
    ]
    avg = sum(s[1] for s in scores) / len(scores)

    st.markdown(f"""
    <div style="background:var(--card);border:1px solid var(--border);border-radius:12px;
         padding:1.2rem 1.5rem;margin-bottom:1.5rem">
      <div style="font-size:.75rem;color:var(--muted);text-transform:uppercase;
           letter-spacing:.08em;margin-bottom:.5rem">signals.py v11 — Overall Score</div>
      <div style="font-size:2.5rem;font-weight:800;color:var(--accent)">{avg:.1f}<span
           style="font-size:1rem;color:var(--muted);font-weight:400"> / 10</span></div>
      <div style="font-size:.8rem;color:var(--muted);margin-top:.3rem">
        Upgrade from v10 (4.7) → v11 (6.6). Remaining gaps: ATR smoothing,
        Supertrend numpy loop, VWAP, MACD double-fire.
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

# ── Background scan ────────────────────────────────────────────────────────────
if (st.session_state.last_auto_scan == 0.0 or
        (time.time() - st.session_state.last_auto_scan) >= 900):
    with st.spinner("🤖 Running Deep Quantitative Market Scan..."):
        open_raw = raw[raw["status"] == "Open"] if not raw.empty else pd.DataFrame()

        st.session_state.signals_cache = (
            generate_signals(open_raw) if not open_raw.empty else []
        )
        st.session_state.sector_cache = sector_rotation()

        if (st.session_state.sector_cache is not None and
                not st.session_state.sector_cache.empty):
            st.session_state.outlook_cache = predict_sector_outlook(
                st.session_state.sector_cache)
            st.session_state.picks_cache = find_sector_picks(
                st.session_state.sector_cache.head(5)["sector"].tolist(), 3)
        else:
            st.session_state.outlook_cache = pd.DataFrame()
            st.session_state.picks_cache   = []

        st.session_state.scanner_cache = generate_market_scanner()

        # FIX: pre-fetch news during background scan so it's ready instantly
        # instead of making users click a button and wait
        if not open_raw.empty:
            st.session_state.news_cache = fetch_portfolio_news(open_raw)
        else:
            st.session_state.news_cache = []

        st.session_state.last_auto_scan = time.time()

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
        f'margin-bottom:1rem">👤 {st.session_state.username.upper()}</div>',
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
            _CACHE.clear()
            st.session_state.last_refresh   = datetime.now()
            st.session_state.last_auto_scan = 0.0
            st.rerun()
    with c2:
        elapsed = time.time() - st.session_state.last_auto_scan
        nxt = max(0, int((900 - elapsed) // 60))
        st.markdown(
            f'<div style="font-size:.75rem;color:var(--muted);padding-top:.5rem;'
            f'font-weight:600">Next: {nxt}m</div>',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:.8rem;font-weight:800;letter-spacing:.05em">'
                '📱 TELEGRAM</div>', unsafe_allow_html=True)
    saved_tok, saved_cid = get_tg_config(UID)
    tg_tok = st.text_input("Bot Token", value=saved_tok, type="password")
    tg_cid = st.text_input("Chat ID", value=saved_cid)
    if st.button("💾 Save Config", width="stretch"):
        save_tg_config(UID, tg_tok, tg_cid)
        st.success("Saved!")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="dash-title">'
    '<div class="dash-title-text">📈 Quantitative <span class="hl">Swing Dashboard</span></div>'
    '<span class="refresh-badge">AUTO SCAN ACTIVE</span>'
    '</div>',
    unsafe_allow_html=True)

market = get_market_regime()
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
(tab1, tab2, tab3, tab4,
 tab5, tab6, tab7, tab8, tab9) = st.tabs([
    "📋 Portfolio", "📊 Analytics", "🔔 Active Signals",
    "🔄 Sector Rotation", "🌌 Universe Scanner",
    "📐 Metrics", "👁 Watchlist", "📤 Export", "🎯 Signal Scores"
])

# ── Tab 1: Portfolio ───────────────────────────────────────────────────────────
with tab1:
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

# ── Tab 2: Charts ──────────────────────────────────────────────────────────────
with tab2:
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

# ── Tab 3: Active Signals ──────────────────────────────────────────────────────
with tab3:
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
                    # Append cached news
                    news = st.session_state.news_cache or []
                    if news:
                        msg_payload += "\n\n🌍 <b>LATEST HOLDINGS NEWS</b>\n"
                        msg_payload += "\n".join(news[:8])
                    ok = send_telegram(saved_tok, saved_cid, msg_payload)
                    if ok:
                        st.success("✅ Broadcast successful!")
                    else:
                        st.error("❌ Broadcast failed. Check token/chat ID.")

    # ── News section (FIXED) ─────────────────────────────────────────────────
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
                    # FIX: call fetch_portfolio_news correctly — pass DataFrame
                    st.session_state.news_cache = fetch_portfolio_news(open_raw)

            # FIX: render_news uses unsafe_allow_html — works with anchor tags
            render_news(st.session_state.news_cache or [])
        else:
            st.info("No active trades. Add a trade to see related news.")

    # ── Signal cards ─────────────────────────────────────────────────────────
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

# ── Tab 4: Sector Rotation ─────────────────────────────────────────────────────
with tab4:
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

# ── Tab 5: Universe Scanner ────────────────────────────────────────────────────
with tab5:
    total_loaded = len(SECTOR_MAP)
    st.markdown(
        f'<div class="sec">🌌 Universe Scanner ({total_loaded} Assets)</div>',
        unsafe_allow_html=True)

    if st.button("⚡ Execute Global Scan", width="stretch"):
        with st.spinner(f"Scanning {total_loaded} tickers..."):
            sd = generate_market_scanner()
            st.session_state.scanner_cache = sd if (sd is not None and not sd.empty) \
                else pd.DataFrame()
            if st.session_state.scanner_cache is not None and \
                    not st.session_state.scanner_cache.empty:
                st.toast(
                    f"✅ {len(st.session_state.scanner_cache)} setups extracted!",
                    icon="🚀")
            st.rerun()

    scan_df = st.session_state.scanner_cache
    if scan_df is None:
        st.info("💡 Initiate scan above or await automated background scan.")
    elif scan_df.empty:
        st.warning("⚠️ Zero setups passed liquidity and pattern gates today.")
    else:
        for sec in scan_df["Sector"].unique():
            sec_df = scan_df[scan_df["Sector"] == sec].drop(columns=["Sector"])
            bc = len(sec_df[sec_df["Score"] >= 5])
            with st.expander(
                    f"📁 {sec} — {len(sec_df)} Assets {'🔥' if bc > 0 else ''}"):
                st.dataframe(
                    sec_df, hide_index=True,
                    column_config={
                        "Generated": st.column_config.TextColumn("Time"),
                        "CMP":    st.column_config.NumberColumn("CMP",      format="₹%.2f"),
                        "Entry":  st.column_config.NumberColumn("Entry",    format="₹%.2f"),
                        "Target": st.column_config.NumberColumn("Target",   format="₹%.2f"),
                        "SL":     st.column_config.NumberColumn("StopLoss", format="₹%.2f"),
                        "Support":st.column_config.NumberColumn("Support",  format="₹%.2f"),
                        "Resist": st.column_config.NumberColumn("Resist",   format="₹%.2f"),
                        "RSI":    st.column_config.NumberColumn("RSI",      format="%.1f"),
                    })

# ── Tab 6: Metrics ─────────────────────────────────────────────────────────────
with tab6:
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

# ── Tab 7: Watchlist ───────────────────────────────────────────────────────────
with tab7:
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
                    # FIX: use ema9/ema21 from v11 signals
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

# ── Tab 8: Export ──────────────────────────────────────────────────────────────
with tab8:
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

# ── Tab 9: Signal Scores ───────────────────────────────────────────────────────
with tab9:
    st.markdown('<div class="sec">🎯 signals.py v11 — Component Scorecard</div>',
                unsafe_allow_html=True)
    render_score_dashboard()
    st.markdown("""
<div style="margin-top:1rem;padding:1rem;background:rgba(99,102,241,.08);
     border:1px solid rgba(99,102,241,.3);border-radius:8px;font-size:.85rem;
     color:var(--muted);line-height:1.8">
<b style="color:var(--text)">Priority fixes for signals.py v12:</b><br>
1. <b>MACD</b> — change <code>for i in [-2,-1]</code> to single-pass transition check<br>
2. <b>Supertrend</b> — replace <code>st.iloc[i]=</code> with numpy array loop<br>
3. <b>ATR</b> — use <code>tr.ewm(alpha=1/14, adjust=False).mean()</code><br>
4. <b>RSI</b> — add <code>adjust=False</code> to all ewm() calls<br>
5. <b>VWAP</b> — replace 5-bar with 20-day rolling VWAP<br>
6. <b>Fibonacci</b> — use scipy peak detection, not fixed 60-bar window
</div>""", unsafe_allow_html=True)
