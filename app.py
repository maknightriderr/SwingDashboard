"""
Swing Trading Portfolio Dashboard v12
Features: Multi-Tenant Secure Login, Password Hashing, User Data Isolation
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
    SECTOR_MAP, _bulk_fetch_history, compute_indicators
)

# ── Auto-refresh config ────────────────────────────────────────────────────────
REFRESH_SEC = 300
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=REFRESH_SEC * 1000, key="dashboard_autorefresh")
except ImportError:
    pass

st.set_page_config(page_title="Swing Dashboard", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# ── Security & Authentication Core ──────────────────────────────────────────────
def make_hash(password):
    """Hashes passwords securely for database storage."""
    return hashlib.sha256(str.encode(password + "swing_salt_99")).hexdigest()

def verify_hash(password, hashed_pw):
    """Verifies an entered password against the database hash."""
    return make_hash(password) == hashed_pw

# ── Multi-Tenant Database Architecture ─────────────────────────────────────────
DB = "trades_v2.db"

def init_db():
    c = sqlite3.connect(DB)
    # Users Table
    c.execute("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT, 
        username TEXT UNIQUE NOT NULL, 
        password_hash TEXT NOT NULL)""")
    # Trades Table (Linked to user_id)
    c.execute("""CREATE TABLE IF NOT EXISTS trades(
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, stock TEXT NOT NULL,
        quantity REAL NOT NULL, buy_at REAL NOT NULL, sell_at REAL,
        status TEXT DEFAULT 'Open', added_date TEXT DEFAULT(date('now')),
        closed_date TEXT)""")
    # Portfolio History (Linked to user_id)
    c.execute("""CREATE TABLE IF NOT EXISTS portfolio_history(
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, snapshot_date TEXT,
        total_invested REAL, current_value REAL)""")
    # Telegram Config (Linked to user_id)
    c.execute("""CREATE TABLE IF NOT EXISTS tg_config(
        user_id INTEGER PRIMARY KEY, bot_token TEXT, chat_id TEXT)""")
    # Watchlist (Linked to user_id)
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

# --- Database User Isolation Functions ---
def register_user(username, password):
    try:
        db("INSERT INTO users(username, password_hash) VALUES(?,?)", (username.lower(), make_hash(password)))
        return True
    except sqlite3.IntegrityError:
        return False # Username taken

def login_user(username, password):
    user = db("SELECT id, password_hash FROM users WHERE username=?", (username.lower(),), fetch=True)
    if user and verify_hash(password, user[0][1]):
        return user[0][0] # Returns user_id
    return None

def get_trades(user_id):
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT * FROM trades WHERE user_id=? ORDER BY id DESC", conn, params=(user_id,))
    conn.close()
    return df

def get_history(user_id):
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT * FROM portfolio_history WHERE user_id=? ORDER BY snapshot_date", conn, params=(user_id,))
    conn.close()
    return df

def get_tg_config(user_id):
    rows = db("SELECT bot_token,chat_id FROM tg_config WHERE user_id=?", (user_id,), fetch=True)
    return rows[0] if rows else ("", "")

def save_tg_config(user_id, token, chat):
    db("INSERT OR REPLACE INTO tg_config(user_id,bot_token,chat_id) VALUES(?,?,?)", (user_id, token, chat))

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
    df = pd.read_sql_query("SELECT * FROM watchlist WHERE user_id=? ORDER BY id DESC", conn, params=(user_id,))
    conn.close()
    return df

def delete_watchlist_item(wid, user_id):
    db("DELETE FROM watchlist WHERE id=? AND user_id=?", (wid, user_id))

# ── Init & Session State Setup ──────────────────────────────────────────────────
init_db()

# Ensure session state variables exist
for k, v in [("user_id", None), ("username", None), ("edit_id", None), ("close_id", None), ("del_id", None),
             ("last_refresh", None), ("last_auto_scan", 0.0), ("sort_col", "stock"), ("sort_asc", False),
             ("signals_cache", None), ("sector_cache", None), ("picks_cache", None),
             ("outlook_cache", None), ("scanner_cache", None), ("filter_status", "All"),
             ("filter_pnl", "All"), ("search", ""), ("theme", "Midnight Pro")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Auth Gatekeeper (Stops execution if not logged in) ──────────────────────────
if st.session_state.user_id is None:
    st.markdown("<h1 style='text-align: center; margin-top: 5rem;'>🔐 Quantitative Swing Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Secure Multi-Tenant Gateway</p>", unsafe_allow_html=True)
    
    _, auth_col2, _ = st.columns([1, 1.5, 1])
    
    with auth_col2:
        tab_login, tab_signup = st.tabs(["Login", "Create Account"])
        
        with tab_login:
            with st.form("login_form"):
                l_user = st.text_input("Username")
                l_pass = st.text_input("Password", type="password")
                
                if st.form_submit_button("Access Terminal", use_container_width=True):
                    if not l_user.strip() or not l_pass.strip():
                        st.error("⚠️ Please fill in all fields.")
                    else:
                        uid = login_user(l_user, l_pass)
                        if uid:
                            st.session_state.user_id = uid
                            st.session_state.username = l_user.strip().lower()
                            st.success("Authenticated. Booting Engine...")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("❌ Invalid Username or Password")
                            
        with tab_signup:
            with st.form("signup_form"):
                s_user = st.text_input("New Username")
                s_pass = st.text_input("New Password", type="password")
                
                if st.form_submit_button("Register Account", use_container_width=True):
                    if len(s_user) < 3 or len(s_pass) < 4:
                        st.error("⚠️ Username > 3 chars and Password > 4 chars required.")
                    else:
                        success = register_user(s_user, s_pass)
                        if success:
                            st.success(f"✅ Account {s_user} registered! Switch to Login tab to enter.")
                        else:
                            st.error("❌ Username already exists.")
                            
    st.stop()  # Halts the rest of the app from rendering while logged out
                        
        with tab_signup:
            with st.form("signup_form"):
                s_user = st.text_input("New Username")
                s_pass = st.text_input("New Password", type="password")
                s_submit = st.form_submit_button("Register Account", width="stretch")
                
                if s_submit:
                    if len(s_user) < 3 or len(s_pass) < 4:
                        st.error("Username > 3 chars and Password > 4 chars required.")
                    else:
                        success = register_user(s_user, s_pass)
                        if success:
                            st.success(f"✅ Account {s_user} registered! You can now log in.")
                        else:
                            st.error("❌ Username already exists.")
    st.stop() # Halts all code execution below this line if not authenticated

# ==============================================================================
# MAIN APPLICATION (Only runs if Authenticated)
# ==============================================================================

# User ID strictly injected into all DB calls below
UID = st.session_state.user_id

THEMES = {
    "Midnight Pro": {
        "bg":"#0b0f19","card":"#111827","input":"#1c212b","border":"#2d3748",
        "text":"#f8fafc","muted":"#94a3b8","green":"#10b981","red":"#ef4444",
        "yellow":"#f59e0b","blue":"#3b82f6","accent":"#6366f1","card2":"#1e293b",
        "gradient":"linear-gradient(135deg, #111827 0%, #1e293b 100%)"
    },
    "Ocean Depth": {
        "bg":"#081229","card":"#0f172a","input":"#1e293b","border":"#334155",
        "text":"#e2e8f0","muted":"#cbd5e1","green":"#059669","red":"#e11d48",
        "yellow":"#d97706","blue":"#0ea5e9","accent":"#0ea5e9","card2":"#0f2242",
        "gradient":"linear-gradient(135deg, #0f172a 0%, #0f2242 100%)"
    },
    "Cyber Neon": {
        "bg":"#050505","card":"#0a0a0a","input":"#141414","border":"#333333",
        "text":"#ffffff","muted":"#a1a1aa","green":"#00ff41","red":"#ff0055",
        "yellow":"#ffe600","blue":"#00e5ff","accent":"#b026ff","card2":"#171717",
        "gradient":"linear-gradient(135deg, #0a0a0a 0%, #171717 100%)"
    }
}

def theme_css(t):
    return f"""
<style>
:root {{
  --bg:{t['bg']}; --card:{t['card']}; --input:{t['input']};
  --border:{t['border']}; --text:{t['text']}; --muted:{t['muted']};
  --green:{t['green']}; --red:{t['red']}; --yellow:{t['yellow']};
  --blue:{t['blue']}; --accent:{t['accent']}; --card2:{t['card2']};
  --gradient:{t['gradient']};
}}

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
    background: var(--bg) !important; background-color: var(--bg) !important; color: var(--text) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}
[data-testid="stHeader"] {{ background: transparent !important; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1rem; padding-bottom: 2rem; max-width: 100%; }}

.dash-title {{ font-size: 1.6rem; font-weight: 800; padding-bottom: 0.8rem; margin-bottom: 1.2rem; border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; }}
.dash-title-text {{ background: -webkit-linear-gradient(45deg, var(--text), var(--muted)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
.dash-title span.hl {{ color: var(--accent); -webkit-text-fill-color: var(--accent); }}

.cards {{ display: flex; gap: 0.8rem; flex-wrap: wrap; margin-bottom: 1.5rem; }}
.card {{ background: var(--gradient); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 1rem; flex: 1; min-width: 140px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); transition: transform 0.2s ease, box-shadow 0.2s ease; }}
.card:hover {{ transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); border-color: rgba(255,255,255,0.1); }}
.card .lbl {{ font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 0.3rem; }}
.card .val {{ font-size: 1.1rem; font-weight: 800; color: var(--text); }}
.card .sub {{ font-size: 0.75rem; color: var(--muted); margin-top: 0.2rem; font-weight: 500; }}

.green {{ color: var(--green) !important; }} .red {{ color: var(--red) !important; }} .yellow {{ color: var(--yellow) !important; }} .blue {{ color: var(--blue) !important; }}
.sec {{ font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text); margin: 1.5rem 0 0.8rem; padding-left: 0.6rem; border-left: 4px solid var(--accent); }}

.tbl-wrap {{ overflow-x: auto; background: var(--card); border: 1px solid var(--border); border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
table.t, .sector-tbl {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; }}
table.t th, .sector-tbl th {{ background: var(--card2); color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700; padding: 0.7rem; text-align: right; border-bottom: 2px solid var(--border); }}
table.t th.l, table.t td.l {{ text-align: left; }}
table.t td, .sector-tbl td {{ padding: 0.6rem 0.7rem; border-bottom: 1px solid var(--border); text-align: right; color: var(--text); }}
table.t tr:last-child td, .sector-tbl tr:last-child td {{ border-bottom: none; }}
table.t tr:hover td, .sector-tbl tr:hover td {{ background: rgba(255,255,255,0.02); }}
table.t tr.row-profit td {{ background: rgba(16, 185, 129, 0.04) !important; }}
table.t tr.row-loss td {{ background: rgba(239, 68, 68, 0.04) !important; }}

.pos {{ color: var(--green); font-weight: 700; }} .neg {{ color: var(--red); font-weight: 700; }} .zero-cell {{ color: var(--muted) !important; }}
.badge {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.65rem; font-weight: 800; text-transform: uppercase; }}
.b-open {{ background: rgba(245,158,11,0.15); color: var(--yellow); border: 1px solid rgba(245,158,11,0.3); }}
.b-cl {{ background: rgba(16,185,129,0.15); color: var(--green); border: 1px solid rgba(16,185,129,0.3); }}
.b-cll {{ background: rgba(239,68,68,0.15); color: var(--red); border: 1px solid rgba(239,68,68,0.3); }}

.sig-grid, .pick-grid, .outlook-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 1rem; margin-top: 0.5rem; }}
.sig-card, .pick-card, .outlook-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1rem; transition: transform 0.2s; }}
.sig-card:hover, .pick-card:hover {{ transform: translateY(-2px); border-color: rgba(255,255,255,0.1); }}
.sig-card.sell {{ border-top: 4px solid var(--red); }} .sig-card.avg {{ border-top: 4px solid var(--yellow); }}
.sig-card.hold {{ border-top: 4px solid var(--green); }} .sig-card.watch {{ border-top: 4px solid var(--muted); }}
.pick-card {{ border-top: 4px solid var(--accent); }}

.sig-action {{ font-size: 0.9rem; font-weight: 800; margin-bottom: 0.4rem; text-transform: uppercase; }}
.sig-meta, .pick-sector {{ font-size: 0.75rem; color: var(--muted); }}
.sig-reason, .pick-prices {{ font-size: 0.8rem; margin-top: 0.5rem; color: var(--text); }}
.sig-price, .pick-reason {{ font-size: 0.75rem; margin-top: 0.6rem; padding-top: 0.6rem; border-top: 1px dashed var(--border); }}
.str-bar {{ height: 4px; border-radius: 2px; margin-top: 0.8rem; background: var(--input); }}
.str-fill {{ height: 100%; border-radius: 2px; transition: width 0.5s; }}

/* BaseWeb Form Fixes */
[data-testid="stSidebar"] {{ background: var(--card) !important; border-right: 1px solid var(--border); padding-top: 2rem; }}
div[data-baseweb="input"], div[data-baseweb="select"], [data-testid="stNumberInputContainer"] {{ background-color: var(--input) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }}
div[data-baseweb="input"] input, [data-testid="stNumberInputContainer"] input {{ color: var(--text) !important; -webkit-text-fill-color: var(--text) !important; background-color: transparent !important; }}
button[data-testid="stNumberInputStepDown"], button[data-testid="stNumberInputStepUp"] {{ background-color: var(--card2) !important; color: var(--text) !important; border: none !important; }}
button[data-testid="stNumberInputStepDown"] svg, button[data-testid="stNumberInputStepUp"] svg, div[data-baseweb="select"] svg {{ fill: var(--text) !important; }}
div[role="listbox"] {{ background-color: var(--card2) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }}
ul[role="listbox"] li {{ color: var(--text) !important; }} ul[role="listbox"] li[aria-selected="true"] {{ background-color: var(--accent) !important; color: white !important; }}

.stButton>button {{ background: var(--card2) !important; border: 1px solid var(--border) !important; color: var(--text) !important; border-radius: 8px !important; font-weight: 600 !important; padding: 0.4rem 1rem !important; transition: all 0.2s ease !important; }}
.stButton>button:hover {{ border-color: var(--accent) !important; background: var(--accent) !important; box-shadow: 0 0 10px rgba(99, 102, 241, 0.4) !important; }}
.stTabs [data-baseweb="tab-list"] {{ background: var(--card); border-bottom: 2px solid var(--border); gap: 1rem; padding: 0 1rem; border-radius: 8px 8px 0 0; }}
.stTabs [data-baseweb="tab"] {{ background: transparent; color: var(--muted); font-weight: 600; padding: 0.8rem 1rem; border: none; border-bottom: 2px solid transparent; }}
.stTabs [aria-selected="true"] {{ background: transparent !important; color: var(--accent) !important; border-bottom-color: var(--accent) !important; }}
[data-testid="stExpander"] {{ background-color: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; margin-bottom: 0.8rem !important; }}
[data-testid="stExpander"] summary p {{ font-weight: 700 !important; color: var(--text) !important; }}
.refresh-badge {{ display: inline-block; background: rgba(16, 185, 129, 0.15); color: var(--green); padding: 0.2rem 0.6rem; border-radius: 6px; font-size: 0.7rem; font-weight: 800; border: 1px solid rgba(16,185,129,0.3); }}
.regime-banner {{ border-radius: 12px; padding: 0.8rem 1.2rem; display: flex; align-items: center; gap: 0.8rem; margin-bottom: 1.5rem; flex-wrap: wrap; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
</style>
"""

# ── Price Fetcher & Logic ───────────────────────────────────────────────────────
_CACHE = {}
_TTL = 300

def fetch_price(symbol):
    key = symbol.upper()
    if key in _CACHE and time.time() - _CACHE[key][1] < _TTL: return _CACHE[key][0]
    for sfx in [".NS", ".BO"]:
        try:
            t = yf.Ticker(key + sfx)
            val = t.fast_info.get("last_price")
            if val is not None and not pd.isna(val): p = round(float(val), 2); _CACHE[key] = (p, time.time()); return p
            h = t.history(period="1d", interval="1d", auto_adjust=True)
            if h is not None and not h.empty and "Close" in h.columns: p = round(float(h["Close"].iloc[-1]), 2); _CACHE[key] = (p, time.time()); return p
        except Exception: continue
    return None

def enrich(df):
    if df.empty: return df
    prices = {s: fetch_price(s) for s in df["stock"].unique()}
    df = df.copy()
    df["cmp"] = df["stock"].map(prices)
    df["nse_label"] = "NSE:" + df["stock"]
    df["invested"] = df["quantity"] * df["buy_at"]
    df["current_amt"] = np.where(df["status"] == "Open", df["quantity"] * df["cmp"].fillna(df["buy_at"]), df["quantity"] * df["sell_at"].fillna(df["buy_at"]))
    df["total_amt"] = np.where(df["sell_at"].notna(), df["quantity"] * df["sell_at"], df["current_amt"])
    df["profit"] = df["total_amt"] - df["invested"]
    df["profit_pct"] = (df["profit"] / df["invested"] * 100).round(2)
    return df

def calc_analytics(df):
    if df.empty: return {}
    closed, open_trades = df[df["status"] == "Closed"], df[df["status"] == "Open"]
    total, total_closed = len(df), len(closed)
    wins = len(closed[closed["profit"] > 0]) if not closed.empty else 0
    losses = len(closed[closed["profit"] < 0]) if not closed.empty else 0
    win_rate = (wins / total_closed * 100) if total_closed > 0 else 0
    avg_win = closed[closed["profit"] > 0]["profit"].mean() if wins > 0 else 0
    avg_loss = abs(closed[closed["profit"] < 0]["profit"].mean()) if losses > 0 else 0
    expectancy = ((wins/total_closed if total_closed>0 else 0) * avg_win) - ((losses/total_closed if total_closed>0 else 0) * avg_loss)
    gp = closed[closed["profit"] > 0]["profit"].sum() if wins > 0 else 0
    gl = abs(closed[closed["profit"] < 0]["profit"].sum()) if losses > 0 else 1
    max_dd = abs((closed["profit"].cumsum() - closed["profit"].cumsum().expanding().max()).min()) if not closed.empty else 0
    avg_hold = round((pd.to_datetime(closed["closed_date"]) - pd.to_datetime(closed["added_date"])).dt.days.mean(), 1) if not closed.empty and "closed_date" in closed.columns else 0
    sharpe = closed["profit_pct"].mean() / closed["profit_pct"].std() if not closed.empty and closed["profit_pct"].std() > 0 else 0
    return {"total_trades": total, "closed_trades": total_closed, "open_trades": len(open_trades), "wins": wins, "losses": losses, "win_rate": round(win_rate, 1), "avg_win": round(avg_win, 0), "avg_loss": round(avg_loss, 0), "expectancy": round(expectancy, 0), "profit_factor": round(gp/gl, 2), "max_drawdown": round(max_dd, 0), "avg_hold_days": avg_hold, "sharpe": round(sharpe, 2)}

def fi(v): return f"₹{v:,.0f}" if not pd.isna(v) else "—"
def fi2(v): return f"₹{v:,.2f}" if not pd.isna(v) else "—"
def fp(v): return f"{'+' if v >= 0 else ''}{v:.2f}%" if not pd.isna(v) else "—"
def cv_cell(v, fn): return f'<td>{fn(v)}</td>' if pd.isna(v) else (f'<td class="profit-cell pos">{fn(v)}</td>' if v > 0 else (f'<td class="profit-cell neg">{fn(v)}</td>' if v < 0 else f'<td class="zero-cell">{fn(v)}</td>'))
def badge(status, profit=None): return '<span class="badge b-open">Open</span>' if status == "Open" else ('<span class="badge b-cll">Closed ✗</span>' if profit is not None and profit < 0 else '<span class="badge b-cl">Closed ✓</span>')
def card(lbl, val, sub="", cls=""): return f'<div class="card"><div class="lbl">{lbl}</div><div class="val {cls}">{val}</div>{"<div class=sub>" + sub + "</div>" if sub else ""}</div>'

def base_layout(fig, title, theme_t=None):
    fig.update_layout(title=dict(text=title, font=dict(size=14, color="#f8fafc", weight="bold"), x=.01), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1", size=11), margin=dict(l=8, r=8, t=45, b=8))
    fig.update_xaxes(gridcolor="#334155", zerolinecolor="#334155", tickfont=dict(color="#cbd5e1"))
    fig.update_yaxes(gridcolor="#334155", zerolinecolor="#334155", tickfont=dict(color="#cbd5e1"))
    return fig

def chart_alloc(df, theme_t=None): return base_layout(go.Figure(go.Pie(labels=df.groupby("stock")["invested"].sum().reset_index()["stock"], values=df.groupby("stock")["invested"].sum().reset_index()["invested"], hole=0.4, marker=dict(colors=px.colors.qualitative.Dark24, line=dict(color="rgba(0,0,0,0)", width=0)), textinfo="percent+label", textfont=dict(size=11, color="#ffffff"))), "Portfolio Allocation", theme_t)
def chart_pnl(df, theme_t=None):
    d = df.sort_values("profit")
    fig = base_layout(go.Figure(go.Bar(x=d["profit"], y=d["stock"], orientation="h", marker=dict(color=["#ef4444" if v < 0 else "#10b981" for v in d["profit"]], line=dict(width=0)), text=[fp(p) for p in d["profit_pct"]], textposition="outside", textfont=dict(color="#f8fafc", size=10))), "P&L by Stock", theme_t)
    fig.update_layout(showlegend=False, margin=dict(l=8, r=55, t=45, b=8)); fig.update_xaxes(tickprefix="₹"); return fig
def chart_donut(df, theme_t=None):
    c = df["status"].value_counts().reset_index(); c.columns = ["Status", "Count"]
    fig = base_layout(go.Figure(go.Pie(labels=c["Status"], values=c["Count"], hole=.6, marker=dict(colors=[{"Open": "#f59e0b", "Closed": "#10b981"}.get(s, "#94a3b8") for s in c["Status"]], line=dict(color="rgba(0,0,0,0)", width=0)), textinfo="percent+value", textfont=dict(size=12, color="#ffffff"))), "Open vs Closed", theme_t)
    fig.add_annotation(text=f"<b>{len(df)}</b><br><span style='font-size:10px'>TRADES</span>", font=dict(size=18, color="#f8fafc"), showarrow=False, x=.5, y=.5); return fig
def chart_growth(hist, cur_val, cur_inv, theme_t=None):
    d = pd.DataFrame(hist[["snapshot_date", "total_invested", "current_value"]].to_dict("records") + ([{"snapshot_date": datetime.now().strftime("%Y-%m-%d"), "total_invested": cur_inv, "current_value": cur_val}] if hist.empty or hist.iloc[-1]["snapshot_date"] != datetime.now().strftime("%Y-%m-%d") else []))
    fig = base_layout(go.Figure([go.Scatter(x=pd.to_datetime(d["snapshot_date"]), y=d["current_value"], name="Value", line=dict(color="#10b981", width=3), fill="tozeroy", fillcolor="rgba(16, 185, 129, 0.1)"), go.Scatter(x=pd.to_datetime(d["snapshot_date"]), y=d["total_invested"], name="Invested", line=dict(color="#3b82f6", width=2, dash="dash"))]), "Portfolio Growth", theme_t)
    fig.update_layout(hovermode="x unified"); fig.update_yaxes(tickprefix="₹"); return fig

def render_signals(signals, theme_t):
    if not signals: st.info("No signals available."); return
    html = '<div class="sig-grid">'
    for s in signals:
        # Safeguard: Ensure 'action' and 'stock' exist
        c = ("sell" if "SELL" in s.get("action","") else "avg" if "AVERAGE" in s.get("action","") else "hold" if "HOLD" in s.get("action","") else "watch")
        clr = theme_t["red"] if c=="sell" else theme_t["yellow"] if c=="avg" else theme_t["green"] if c=="hold" else theme_t["muted"]
        
        # Safe access using .get()
        stock_name = s.get('stock', 'N/A')
        action = s.get('action', 'SIGNAL')
        
        ph = (f"🎯 Exit: ₹{s.get('target','—')} | 🛑 Re-entry: ₹{s.get('stop_loss','—')}<br>📉 {s.get('trend','—')} | MACD: {s.get('macd_signal','—')}" if c=="sell" else f"💰 Avg: ₹{s.get('avg_price','—')} | New Avg: ₹{s.get('new_avg','—')}<br>🛑 SL: ₹{s.get('new_sl','—')} | 🎯 Target: ₹{s.get('target','—')}" if c=="avg" else f"🎯 Target: ₹{s.get('target','—')} | 🛑 SL: ₹{s.get('stop_loss','—')}<br>📊 R:R {s.get('risk_reward','—')} | {s.get('trend','—')}")
        
        html += f"""<div class="sig-card {c}"><div class="sig-action" style="color:{clr}">{action}</div><div style="font-size:.9rem;font-weight:800;margin-bottom:.3rem">{stock_name} <span class="nse-lbl">{s.get('sector','')}</span></div><div class="sig-meta">CMP ₹{s.get('cmp','—')} · RSI {s.get('rsi','—')} · {s.get('pct_from_buy',0):+.1f}%</div><div class="sig-reason">{s.get('reason','')}</div><div class="sig-price">{ph}</div><div class="str-bar"><div class="str-fill" style="width:{s.get('strength',30)}%;background:{clr}"></div></div></div>"""
    st.markdown(html + "</div>", unsafe_allow_html=True)

def render_sector(sdf, t):
    if sdf is None or sdf.empty: return
    rows = "".join([f"<tr><td style='font-weight:700'>{'🥇' if r['rank']==1 else '🥈' if r['rank']==2 else '🥉' if r['rank']==3 else '📊'} #{int(r['rank'])}</td><td><b style='font-size:0.9rem'>{r['sector']}</b></td><td style='color:var(--muted);font-size:.75rem'>{r['stocks']}</td><td style='text-align:center;font-weight:600'>{r['avg_rsi']:.0f}</td><td style='text-align:right'><span class='{'pos' if r['avg_pct']>0 else 'neg'}'>{r['avg_pct']:+.1f}%</span></td><td><div style='background:{t['input']};height:6px;width:100%;border-radius:4px'><div style='background:{t['accent']};width:{min(r['momentum_score']*100,100):.0f}%;height:6px;border-radius:4px'></div></div><span style='font-size:.75rem;font-weight:600'>{r['momentum_score']:.2f}</span></td></tr>" for _, r in sdf.iterrows()])
    st.markdown(f'<div class="tbl-wrap"><table class="sector-tbl"><thead><tr><th>Rank</th><th>Sector</th><th>Top Movers</th><th style="text-align:center">Avg RSI</th><th style="text-align:right">Chg</th><th>Momentum</th></tr></thead><tbody>{rows}</tbody></table></div>', unsafe_allow_html=True)

def render_outlook(odf, t):
    if odf is None or odf.empty: return
    st.markdown('<div class="outlook-grid">' + "".join([f"<div class='outlook-card'><div class='outlook-sector'>{r['sector']}</div><div class='outlook-label' style='color:{t['green'] if 'Bullish' in r['outlook'] else t['red']}'>{r['outlook']}</div><div class='outlook-meta'>Conf: {r['confidence']}% · Mom: {r['momentum']:.2f}<br>RSI: {r['avg_rsi']:.0f} · Chg: {r['avg_pct']:+.1f}%</div></div>" for _, r in odf.iterrows()]) + "</div>", unsafe_allow_html=True)

def render_picks(picks, t):
    if not picks: return
    st.markdown('<div class="pick-grid">' + "".join([f"<div class='pick-card' style='border-top-color:{t['green'] if p['score']>=70 else t['yellow'] if p['score']>=55 else t['muted']}'><div class='pick-stock'>{p['stock']} <span class='pick-sector'>{p['sector']}</span></div><div style='font-size:.8rem;color:{t['muted']};font-weight:600;margin-top:3px'>CMP ₹{p['cmp']} · RSI {p['rsi']} · {p['trend']}</div><div class='pick-prices'>🎯 Entry: ₹{p['entry']}<br>🚀 Target: ₹{p['target']}<br>🛑 SL: ₹{p['stop_loss']}<br>📊 R:R: {p['risk_reward']} · Score: {p['score']}</div><div class='pick-reason'>{p['reason']}</div></div>" for p in picks]) + "</div>", unsafe_allow_html=True)

# ── Load Data (User Scoped) ───────────────────────────────────────────────────
raw = get_trades(UID)
df = enrich(raw) if not raw.empty else raw.copy()

if st.session_state.last_refresh is None or (datetime.now() - st.session_state.last_refresh).seconds >= _TTL:
    st.session_state.last_refresh = datetime.now()

# ── 🤖 Deep Background Scan ───────────────────────────────────────────────────
if st.session_state.last_auto_scan == 0.0 or (time.time() - st.session_state.last_auto_scan) >= 900:
    with st.spinner("🤖 Running Deep Quantitative Market Scan..."):
        st.session_state.signals_cache = generate_signals(raw[raw["status"] == "Open"]) if not df[df["status"] == "Open"].empty else []
        st.session_state.sector_cache = sector_rotation()
        if st.session_state.sector_cache is not None and not st.session_state.sector_cache.empty:
            st.session_state.outlook_cache = predict_sector_outlook(st.session_state.sector_cache)
            st.session_state.picks_cache = find_sector_picks(st.session_state.sector_cache.head(5)["sector"].tolist(), 3)
        else:
            st.session_state.outlook_cache = pd.DataFrame(); st.session_state.picks_cache = []
        st.session_state.scanner_cache = generate_market_scanner()
        st.session_state.last_auto_scan = time.time()

# ── Metrics Calc ─────────────────────────────────────────────────────────────
if not df.empty:
    odf, cdf = df[df["status"]=="Open"], df[df["status"]=="Closed"]
    t_inv, t_cur, t_real, t_unreal = df["invested"].sum(), df["current_amt"].sum(), cdf["profit"].sum() if not cdf.empty else 0, odf["profit"].sum() if not odf.empty else 0
    t_pnl = df["profit"].sum(); t_pnl_pct = t_pnl/t_inv*100 if t_inv>0 else 0
    best, worst = df.loc[df["profit_pct"].idxmax(),"stock"], df.loc[df["profit_pct"].idxmin(),"stock"]
    save_snapshot(UID, t_inv, t_cur)
else:
    odf = cdf = pd.DataFrame(); t_inv=t_cur=t_real=t_unreal=t_pnl=t_pnl_pct=0; best=worst="—"

theme_t = THEMES[st.session_state.theme]
st.markdown(theme_css(theme_t), unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<div style="font-size:0.85rem;font-weight:800;color:var(--accent);margin-bottom:1rem;">👤 User: {st.session_state.username.upper()}</div>', unsafe_allow_html=True)
    if st.button("🚪 Logout", width="stretch"):
        st.session_state.clear()
        st.rerun()
        
    st.markdown("<hr style='margin:1rem 0; border-color:var(--border)'>", unsafe_allow_html=True)
    
    st.markdown('<div style="font-size:0.8rem;font-weight:800;letter-spacing:0.05em">🎨 UI THEME</div>', unsafe_allow_html=True)
    if (new_theme := st.selectbox("Theme", list(THEMES.keys()), index=list(THEMES.keys()).index(st.session_state.theme), label_visibility="collapsed")) != st.session_state.theme:
        st.session_state.theme = new_theme; st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.1rem;font-weight:800;color:var(--accent);margin-bottom:0.8rem">⚡ Trade Entry</div>', unsafe_allow_html=True)
    
    em = st.session_state.edit_id is not None
    erow = raw[raw["id"]==st.session_state.edit_id].iloc[0] if em and not raw.empty else None
    if em: st.markdown('<div style="background:rgba(99,102,241,0.15);border:1px solid rgba(99,102,241,0.4); border-radius:8px;padding:0.5rem;font-size:0.8rem;color:var(--accent);margin-bottom:1rem;font-weight:700">✏️ Editing trade</div>', unsafe_allow_html=True)

    with st.form("trade_form", clear_on_submit=True):
        s_in = st.text_input("Stock Symbol", value=erow["stock"] if erow is not None else "", placeholder="CDSL, IRFC…")
        q_in = st.number_input("Quantity", min_value=1, step=1, value=int(erow["quantity"]) if erow is not None else 1)
        b_in = st.number_input("Buy At ₹", min_value=0.01, step=0.05, value=float(erow["buy_at"]) if erow is not None else 0.01, format="%.2f")
        sel_in = st.number_input("Sell At ₹ (optional)", min_value=0.0, step=0.05, value=float(erow["sell_at"]) if (erow is not None and erow["sell_at"]) else 0.0, format="%.2f")
        if st.form_submit_button("💾 Update Trade" if em else "➕ Execute Entry", width="stretch"):
            if not s_in.strip(): st.error("Symbol required")
            elif b_in <= 0: st.error("Buy price must be > 0")
            else:
                sv = sel_in if sel_in > 0 else None
                if em: update_trade(st.session_state.edit_id, UID, s_in, q_in, b_in, sv, "Closed" if sv else "Open"); st.session_state.edit_id = None; st.success("Updated!")
                else: add_trade(UID, s_in, q_in, b_in, sv); st.success(f"Added {s_in.upper()}")
                _CACHE.clear(); st.session_state.last_auto_scan = 0.0; st.rerun()

    if em and st.button("✖ Cancel Edit", width="stretch"): st.session_state.edit_id = None; st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.8rem;font-weight:800;letter-spacing:0.05em">🔍 FILTERS</div>', unsafe_allow_html=True)
    st.session_state.filter_status = st.selectbox("Status", ["All", "Open", "Closed"], index=["All", "Open", "Closed"].index(st.session_state.filter_status), label_visibility="collapsed")
    st.session_state.filter_pnl = st.selectbox("P&L", ["All", "Profitable", "Loss"], index=["All", "Profitable", "Loss"].index(st.session_state.filter_pnl), label_visibility="collapsed")
    st.session_state.search = st.text_input("Search", value=st.session_state.search, placeholder="Search symbol…", label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Force Scan", width="stretch"): _CACHE.clear(); st.session_state.last_refresh = datetime.now(); st.session_state.last_auto_scan = 0.0; st.rerun()
    with c2: st.markdown(f'<div style="font-size:0.75rem;color:var(--muted);padding-top:0.5rem;font-weight:600">Next: {max(0, int((900 - (time.time() - st.session_state.last_auto_scan)) // 60))}m</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.8rem;font-weight:800;letter-spacing:0.05em">📱 TELEGRAM</div>', unsafe_allow_html=True)
    saved_tok, saved_cid = get_tg_config(UID)
    tg_tok = st.text_input("Bot Token", value=saved_tok, type="password")
    tg_cid = st.text_input("Chat ID", value=saved_cid)
    if st.button("💾 Save Config", width="stretch"): save_tg_config(UID, tg_tok, tg_cid); st.success("Saved!")

# ── Main Header ────────────────────────────────────────────────────────────────
st.markdown(f'<div class="dash-title"><div class="dash-title-text">📈 Quantitative <span class="hl">Swing Dashboard</span></div> <span class="refresh-badge">AUTO SCAN ACTIVE</span></div>', unsafe_allow_html=True)

market = get_market_regime()
rc_bg, rc_clr, rc_border = {"Strong Bull": ("rgba(16,185,129,0.15)", "#10b981", "border: 1px solid rgba(16,185,129,0.4)"), "Bull": ("rgba(16,185,129,0.1)", "#10b981", "border: 1px solid rgba(16,185,129,0.2)"), "Bull Pullback": ("rgba(245,158,11,0.15)", "#f59e0b", "border: 1px solid rgba(245,158,11,0.4)"), "Strong Bear": ("rgba(239,68,68,0.15)", "#ef4444", "border: 1px solid rgba(239,68,68,0.4)"), "Bear": ("rgba(239,68,68,0.1)", "#ef4444", "border: 1px solid rgba(239,68,68,0.2)"), "Bear Rally": ("rgba(245,158,11,0.15)", "#f59e0b", "border: 1px solid rgba(245,158,11,0.4)")}.get(market["regime"], ("rgba(148,163,184,0.1)", "#94a3b8", "border: 1px solid rgba(148,163,184,0.3)"))
indices_html = "".join([f'<span style="color:var(--text);font-size:0.8rem;padding:0 0.8rem;border-right:1px solid rgba(255,255,255,0.1)">{name} <b>{f"₹{d['price']:,.0f}" if d.get("price") else "—"}</b> <span style="color:{"var(--red)" if (name=="India VIX" and d.get("chg_pct",0)>0) else "var(--green)" if (name=="India VIX" and d.get("chg_pct",0)<0) else "var(--green)" if d.get("chg_pct",0)>0 else "var(--red)"};font-weight:700">{d.get("chg_pct",0):+.2f}%</span></span>' for name, d in market.get("indices", {}).items()])
st.markdown(f'<div class="regime-banner" style="background:{rc_bg};{rc_border};backdrop-filter:blur(10px);"><span style="color:{rc_clr};font-weight:800;font-size:0.9rem;white-space:nowrap;letter-spacing:0.05em">🌐 {market["regime"].upper()} (CONF: {market.get("confidence", "—")}%)</span>{indices_html}<span style="color:var(--muted);font-size:0.75rem;white-space:nowrap;padding-left:0.5rem;font-weight:600">SUP: {f"₹{market.get("support"):,.0f}" if market.get("support") else "—"} | RES: {f"₹{market.get("resistance"):,.0f}" if market.get("resistance") else "—"} | RSI {market.get("nifty_rsi", "—")} | RISK: {market.get("risk_level","—")}</span></div>', unsafe_allow_html=True)

pnl_c = "green" if t_pnl >= 0 else "red"
r_c = "green" if t_real >= 0 else "red"
u_c = "green" if t_unreal >= 0 else "red"

st.markdown('<div class="cards">' + card("Total Invested", fi(t_inv), "", "blue") + card("Portfolio Value", fi(t_cur), "", "blue") + card("Total P&L", fi(t_pnl), fp(t_pnl_pct), pnl_c) + card("Realized P&L", fi(t_real), "", r_c) + card("Unrealized P&L", fi(t_unreal), "", u_c) + card("Open Trades", str(len(odf)), "Active", "yellow") + card("Closed Trades", str(len(cdf)), "Historical", "green" if len(cdf)>0 else "") + card("Best Trade 🏆", best, "", "green") + card("Worst Trade 📉", worst, "", "red") + '</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["📋 Portfolio", "📊 Analytics Charts", "🔔 Active Signals", "🔄 Sector Rotation", "🌌 Universe Scanner", "📐 Metrics", "👁 Watchlist", "📤 Export Data"])

with tab1:
    if df.empty: st.info("No trades yet. Use the sidebar to execute an entry.")
    else:
        fdf = df.copy()
        if st.session_state.filter_status != "All": fdf = fdf[fdf["status"] == st.session_state.filter_status]
        if st.session_state.filter_pnl == "Profitable": fdf = fdf[fdf["profit"] > 0]
        elif st.session_state.filter_pnl == "Loss": fdf = fdf[fdf["profit"] < 0]
        if st.session_state.search.strip(): fdf = fdf[fdf["stock"].str.upper().str.contains(st.session_state.search.upper())]
        sort_opts = {"Stock": "stock", "Qty": "quantity", "Buy At": "buy_at", "CMP": "cmp", "Invested": "invested", "P&L ₹": "profit", "P&L %": "profit_pct"}
        sc1, sc2 = st.columns([3, 1])
        with sc1: sort_col = sort_opts[st.selectbox("Sort by", list(sort_opts.keys()), index=list(sort_opts.values()).index(st.session_state.sort_col) if st.session_state.sort_col in sort_opts.values() else 0, label_visibility="collapsed")]
        with sc2: asc = st.toggle("⬆ Ascending", value=st.session_state.sort_asc); st.session_state.sort_asc = asc
        st.session_state.sort_col = sort_col
        if sort_col in fdf.columns: fdf = fdf.sort_values(sort_col, ascending=asc, na_position="last")

        st.markdown(f'<div class="sec">Open Positions & History ({len(fdf)})</div>', unsafe_allow_html=True)
        rows = "".join([f"<tr class='{'row-profit' if r.get('profit',0)>0 else 'row-loss' if r.get('profit',0)<0 else 'row-neutral'}'><td class='l'><span class='nse-lbl'>{r.get('nse_label','')}</span></td><td class='l'><b style='font-size:0.9rem'>{r['stock']}</b><br><span class='nse-lbl'>{get_sector(r['stock'])} · {r.get('added_date','')}</span></td><td>{int(r['quantity'])}</td><td>{fi2(r['buy_at'])}</td>{'<td class=zero-cell>—</td>' if pd.isna(r.get('cmp')) else f'<td class=pos>{fi2(r["cmp"])}</td>' if r['cmp']>r['buy_at'] else f'<td class=neg>{fi2(r["cmp"])}</td>' if r['cmp']<r['buy_at'] else f'<td>{fi2(r["cmp"])}</td>'}<td>{fi2(r['sell_at']) if pd.notna(r.get('sell_at')) else '—'}</td><td>{fi(r['invested'])}</td>{'<td>—</td>' if pd.isna(r.get('current_amt',0)) else f'<td class=pos>{fi(r["current_amt"])}</td>' if r['current_amt']>r['invested'] else f'<td class=neg>{fi(r["current_amt"])}</td>' if r['current_amt']<r['invested'] else f'<td>{fi(r["current_amt"])}</td>'}{cv_cell(r.get('profit',0), fi)}{cv_cell(r.get('profit_pct',0), fp)}<td>{badge(r['status'], r.get('profit',0))}</td></tr>" for _, r in fdf.iterrows()])
        st.markdown(f'<div class="tbl-wrap"><table class="t"><thead><tr><th class="l">NSE</th><th class="l">Asset</th><th>Qty</th><th>Entry</th><th>CMP</th><th>Exit</th><th>Invested</th><th>Value</th><th>P&L ₹</th><th>P&L %</th><th>Status</th></tr></thead><tbody>{rows}</tbody></table></div>', unsafe_allow_html=True)

        st.markdown('<div class="sec">Manage Positions</div>', unsafe_allow_html=True)
        opts = [f"{r['id']} — {r['stock']}" for _, r in fdf.iterrows()]
        if opts:
            ca, cb, cc, cd = st.columns([3, 1, 1, 1])
            with ca: sel_id = int(st.selectbox("Select Trade ID", opts, label_visibility="collapsed").split(" — ")[0])
            with cb: 
                if st.button("✏️ Modify", width="stretch"): st.session_state.edit_id = sel_id; st.rerun()
            with cc: 
                if st.button("🔒 Close Pos", width="stretch"): st.session_state.close_id = sel_id; st.rerun()
            with cd: 
                if st.button("🗑 Drop", width="stretch"): st.session_state.del_id = sel_id; st.rerun()

        if st.session_state.close_id:
            st.markdown("---"); st.markdown("**Execute Close — Confirm Exit Price**")
            sp = st.number_input("Exit Price ₹", min_value=0.01, step=0.05, format="%.2f")
            x1, x2 = st.columns(2)
            with x1: 
                if st.button("✅ Confirm Exit", width="stretch"): close_trade(st.session_state.close_id, UID, sp); st.session_state.close_id = None; st.rerun()
            with x2: 
                if st.button("✖ Abort", width="stretch"): st.session_state.close_id = None; st.rerun()

        if st.session_state.del_id:
            st.markdown("---"); st.warning(f"Drop trade ID #{st.session_state.del_id} from database? This is irreversible.")
            y1, y2 = st.columns(2)
            with y1: 
                if st.button("🗑 Confirm Drop", width="stretch"): delete_trade(st.session_state.del_id, UID); st.session_state.del_id = None; st.rerun()
            with y2: 
                if st.button("✖ Abort", width="stretch"): st.session_state.del_id = None; st.rerun()

with tab2:
    if df.empty: st.info("Execute trades to populate visualization models.")
    else:
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(chart_alloc(df, theme_t))
        with c2: st.plotly_chart(chart_donut(df, theme_t))
        st.plotly_chart(chart_pnl(df, theme_t))
        st.plotly_chart(chart_growth(get_history(UID), t_cur, t_inv, theme_t))

with tab3:
    st.markdown('<div class="sec">Active Portfolio Signals & Risk Management</div>', unsafe_allow_html=True)
    s1, s2 = st.columns([2, 1])
    with s1: st.caption("🤖 Neural background scan refreshes signal intelligence every 15 minutes.")
    with s2:
        if st.button("📲 Push to Telegram", width="stretch", disabled=not bool(saved_tok and saved_cid)):
            if st.session_state.signals_cache is not None:
                ok = send_telegram(saved_tok, saved_cid, build_telegram_message(st.session_state.signals_cache, st.session_state.sector_cache if st.session_state.sector_cache is not None else pd.DataFrame(), st.session_state.picks_cache if st.session_state.picks_cache is not None else []))
                st.success("✅ Broadcast successful!") if ok else st.error("❌ Broadcast failed.")
    if st.session_state.signals_cache is not None:
        nc = {"SELL": 0, "AVERAGE": 0, "HOLD": 0, "WATCH": 0}
        for s in st.session_state.signals_cache:
            for k in nc:
                if k in s.get("action", ""): nc[k] += 1
        st.markdown(f'<div style="display:flex;gap:.8rem;margin:.5rem 0 1rem"><span style="background:rgba(239,68,68,0.15);color:#ef4444;padding:.3rem .8rem;border-radius:6px;font-size:.8rem;font-weight:800;border:1px solid rgba(239,68,68,0.3)">🔴 SELL: {nc["SELL"]}</span><span style="background:rgba(245,158,11,0.15);color:#f59e0b;padding:.3rem .8rem;border-radius:6px;font-size:.8rem;font-weight:800;border:1px solid rgba(245,158,11,0.3)">🟡 AVERAGE: {nc["AVERAGE"]}</span><span style="background:rgba(16,185,129,0.15);color:#10b981;padding:.3rem .8rem;border-radius:6px;font-size:.8rem;font-weight:800;border:1px solid rgba(16,185,129,0.3)">🟢 HOLD: {nc["HOLD"]}</span></div>', unsafe_allow_html=True)
        render_signals(st.session_state.signals_cache, theme_t)

with tab4:
    st.markdown('<div class="sec">Macro Sector Rotation & Capital Flow</div>', unsafe_allow_html=True)
    if st.session_state.sector_cache is not None:
        render_sector(st.session_state.sector_cache, theme_t)
        if not st.session_state.sector_cache.empty:
            top = st.session_state.sector_cache.iloc[0]
            st.markdown(f'<div style="margin-top:1rem;background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.3);border-radius:8px;padding:0.8rem 1rem;font-size:0.85rem">🥇 <b style="color:var(--text)">Leading Sector: {top["sector"]}</b> — Momentum {top["momentum_score"]:.2f} | Avg RSI {top["avg_rsi"]:.0f} | Flow {top["avg_pct"]:+.1f}%<br><span style="color:var(--muted);font-size:0.75rem;margin-top:5px;display:block">Constituents tracking positive: {top["stocks"]}</span></div>', unsafe_allow_html=True)
    if st.session_state.outlook_cache is not None and not st.session_state.outlook_cache.empty:
        st.markdown('<div class="sec" style="margin-top:2rem">📈 Institutional Outlook Prediction</div>', unsafe_allow_html=True)
        render_outlook(st.session_state.outlook_cache, theme_t)
    if st.session_state.picks_cache is not None:
        st.markdown('<div class="sec" style="margin-top:2rem">🎯 Algorithmic Entry Setups</div>', unsafe_allow_html=True)
        render_picks(st.session_state.picks_cache, theme_t)

with tab5:
    total_loaded = len(SECTOR_MAP)
    st.markdown(f'<div class="sec">🌌 Global Market Confluence Scanner ({total_loaded} Assets)</div>', unsafe_allow_html=True)
    if st.button("⚡ Execute Global Universe Scan", width="stretch"):
        with st.spinner(f"Initiating multi-threaded pattern recognition across {total_loaded} tickers..."):
            sd = generate_market_scanner()
            if sd is not None and not sd.empty: st.session_state.scanner_cache = sd; st.toast(f"✅ Extracted {len(sd)} institutional-grade setups!", icon="🚀")
            else: st.session_state.scanner_cache = pd.DataFrame()
            st.rerun()
    
    scan_df = st.session_state.scanner_cache
    if scan_df is None: st.info("💡 Standby. Initiate scan above or await automated background interval.")
    elif scan_df.empty: st.warning("⚠️ Zero setups passed the strict liquidity and pattern gates today. Market conditions may be hostile.")
    else:
        for sec in scan_df["Sector"].unique():
            sec_df = scan_df[scan_df["Sector"] == sec].drop(columns=["Sector"])
            bc = len(sec_df[sec_df["Score"] >= 4])
            with st.expander(f"📁 {sec} — {len(sec_df)} Assets {'🔥' if bc > 0 else ''}"):
                st.dataframe(sec_df, hide_index=True, column_config={"Generated": st.column_config.TextColumn("Time"), "CMP": st.column_config.NumberColumn("CMP", format="₹%.2f"), "Entry": st.column_config.NumberColumn("Entry", format="₹%.2f"), "Target": st.column_config.NumberColumn("Target", format="₹%.2f"), "SL": st.column_config.NumberColumn("StopLoss", format="₹%.2f"), "Support": st.column_config.NumberColumn("Support", format="₹%.2f"), "Resist": st.column_config.NumberColumn("Resist", format="₹%.2f"), "RSI": st.column_config.NumberColumn("RSI", format="%.1f")})

with tab6:
    a = calc_analytics(df)
    if not a or a.get("closed_trades", 0) == 0: st.info("Metrics require historical closed trades to calculate probabilities.")
    else:
        st.markdown('<div class="sec">Strategy Performance Metrics</div>', unsafe_allow_html=True)
        st.markdown('<div class="cards">' + card("Win Rate", f'{a["win_rate"]}%', f'{a["wins"]}W / {a["losses"]}L', "green" if a["win_rate"] >= 50 else "red") + card("Profit Factor", str(a["profit_factor"]), "Gross P / Gross L") + card("Expectancy", f'₹{a["expectancy"]}') + card("Avg Win", f'₹{a["avg_win"]:,.0f}') + card("Avg Loss", f'₹{a["avg_loss"]:,.0f}', "", "red") + card("Max Drawdown", f'₹{a["max_drawdown"]:,.0f}', "", "red") + card("Avg Hold", f'{a["avg_hold_days"]}d') + card("Sharpe", str(a["sharpe"])) + '</div>', unsafe_allow_html=True)

with tab7:
    st.markdown('<div class="sec">👁 Target Watchlist</div>', unsafe_allow_html=True)
    def drop_watchlist_cb(w_id, s_name):
        delete_watchlist_item(w_id, UID)
        st.toast(f"🗑️ Dropped {s_name} from database!")

    with st.form(key="add_stock_form", clear_on_submit=True):
        col_input, col_btn = st.columns([4, 1])
        with col_input: new_stock = st.text_input("Stock Ticker", placeholder="e.g., SBIN, TATAMOTORS", label_visibility="collapsed").upper().strip()
        with col_btn: submit_btn = st.form_submit_button("➕ Add Stock", width="stretch")
        if submit_btn and new_stock:
            add_watchlist(UID, new_stock)
            st.toast(f"🚀 {new_stock} saved to database!"); st.rerun()

    wdf = get_watchlist(UID)
    if not wdf.empty:
        st.markdown('<div class="sec" style="margin-top:1rem;">Live Monitored Assets</div>', unsafe_allow_html=True)
        wl_symbols = wdf['stock'].tolist()
        with st.spinner("Fetching live metrics for watchlist..."): wl_data = _bulk_fetch_history(wl_symbols, period="3mo")
        cols = st.columns(3)
        for i, row in wdf.iterrows():
            stock = row['stock']; wid = int(row['id']); col = cols[i % 3]
            with col:
                with st.container():
                    df_hist = wl_data.get(stock)
                    ind = compute_indicators(stock, period="3mo", prefetched_df=df_hist)
                    if ind:
                        cmp_v, rsi_v, trend, sup, res = ind.get('cmp', '—'), ind.get('rsi', '—'), ind.get('trend', '—'), ind.get('support', '—'), ind.get('resistance', '—')
                        brd_color = theme_t['green'] if "Uptrend" in trend else (theme_t['red'] if "Downtrend" in trend else theme_t['yellow'])
                        st.markdown(f"""<div style="background:var(--card); border-top:4px solid {brd_color}; border-radius:8px; padding:1rem; box-shadow:0 4px 6px rgba(0,0,0,0.05); margin-bottom: 0.5rem;"><div style="font-size:1.1rem; font-weight:800; color:var(--text);">{stock}</div><div style="font-size:0.75rem; color:var(--muted); margin-bottom:0.5rem; text-transform:uppercase;">{get_sector(stock)}</div><div style="font-size:0.8rem; line-height:1.6; color:var(--text);"><b>CMP:</b> ₹{cmp_v}<br><b>RSI:</b> {rsi_v} | <b>Trend:</b> {trend}<br><b>Sup:</b> ₹{sup} | <b>Res:</b> ₹{res}</div></div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""<div style="background:var(--card); border-top:4px solid var(--muted); border-radius:8px; padding:1rem; box-shadow:0 4px 6px rgba(0,0,0,0.05); margin-bottom: 0.5rem;"><div style="font-size:1.1rem; font-weight:800; color:var(--text);">{stock}</div><div style="font-size:0.85rem; color:var(--red); margin-bottom:0.5rem;">Market Data Unavailable</div></div>""", unsafe_allow_html=True)
                    st.button("🗑️ Drop Stock", key=f"wl_del_{wid}", on_click=drop_watchlist_cb, args=(wid, stock), width="stretch")
    else: st.info("Your watchlist is currently empty. Enter a ticker symbol above to start tracking.")

with tab8:
    if df.empty: st.info("No data available for export.")
    else:
        st.markdown('<div class="sec">Raw Database Export</div>', unsafe_allow_html=True)
        st.dataframe(df)
