"""
Swing Trading Portfolio Dashboard v8
Features: Universe Risk Metrics, Sensex Integration, Safe Signals Processing
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io, time

from signals import (
    generate_signals, sector_rotation, predict_sector_outlook,
    find_sector_picks, send_telegram, build_telegram_message,
    get_sector, get_market_regime, generate_market_scanner,
    SECTOR_MAP  
)

# ── Auto-refresh every 5 minutes for UI, background scan runs every 15 mins ───
REFRESH_SEC = 300
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=REFRESH_SEC * 1000, key="dashboard_autorefresh")
except ImportError:
    pass

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Swing Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme definitions ─────────────────────────────────────────────────────────
THEMES = {
    "Midnight": {
        "bg":"#0d1117","card":"#161b22","input":"#21262d","border":"#30363d",
        "text":"#e6edf3","muted":"#8b949e","green":"#3fb950","red":"#f85149",
        "yellow":"#d29922","blue":"#388bfd","accent":"#58a6ff","card2":"#1c2128",
    },
    "Ocean": {
        "bg":"#0a192f","card":"#112240","input":"#1d3461","border":"#233554",
        "text":"#ccd6f6","muted":"#8892b0","green":"#64ffda","red":"#ff6b6b",
        "yellow":"#ffd93d","blue":"#64ffda","accent":"#64ffda","card2":"#172a45",
    },
    "Cyberpunk": {
        "bg":"#0a0a0a","card":"#1a1a2e","input":"#16213e","border":"#0f3460",
        "text":"#e0e0e0","muted":"#8b949e","green":"#00ff41","red":"#ff0054",
        "yellow":"#f5d300","blue":"#00d4ff","accent":"#00ff41","card2":"#1f1f38",
    },
    "Light": {
        "bg":"#f6f8fa","card":"#ffffff","input":"#e1e4e8","border":"#d1d5da",
        "text":"#24292e","muted":"#586069","green":"#28a745","red":"#d73a49",
        "yellow":"#e36209","blue":"#0366d6","accent":"#0366d6","card2":"#f0f0f0",
    },
    "Forest": {
        "bg":"#0b1a0b","card":"#142814","input":"#1e3a1e","border":"#2a4a2a",
        "text":"#c8e6c9","muted":"#81c784","green":"#4caf50","red":"#ef5350",
        "yellow":"#fdd835","blue":"#4fc3f7","accent":"#66bb6a","card2":"#1a3a1a",
    },
}

def theme_css(t):
    return f"""
<style>
:root {{
  --bg:{t['bg']};--card:{t['card']};--input:{t['input']};
  --border:{t['border']};--text:{t['text']};--muted:{t['muted']};
  --green:{t['green']};--red:{t['red']};--yellow:{t['yellow']};
  --blue:{t['blue']};--accent:{t['accent']};--card2:{t['card2']};
}}
html,body,[class*="css"]{{background:var(--bg);color:var(--text);
  font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif}}
#MainMenu,footer,header{{visibility:hidden}}
.block-container{{padding-top:.75rem;padding-bottom:2rem;max-width:100%}}

.dash-title{{font-size:1.4rem;font-weight:700;letter-spacing:-.5px;
  border-bottom:1px solid var(--border);padding-bottom:.6rem;margin-bottom:1rem}}
.dash-title span{{color:var(--accent)}}

.cards{{display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:1rem}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:8px;
  padding:.7rem .85rem;flex:1;min-width:120px;max-width:200px;overflow:hidden}}
.card .lbl{{font-size:.6rem;text-transform:uppercase;letter-spacing:.07em;
  color:var(--muted);margin-bottom:.2rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.card .val{{font-size:.95rem;font-weight:700;white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis;color:var(--text)}}
.card .sub{{font-size:.7rem;color:var(--muted);margin-top:.1rem;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.green{{color:var(--green)!important}} .red{{color:var(--red)!important}}
.yellow{{color:var(--yellow)!important}} .blue{{color:var(--accent)!important}}

.sec{{font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;
  color:var(--muted);margin:1rem 0 .4rem;border-left:3px solid var(--accent);padding-left:.5rem}}

.tbl-wrap{{overflow-x:auto;background:var(--card);border:1px solid var(--border);border-radius:8px}}
table.t{{width:100%;border-collapse:collapse;font-size:.76rem}}
table.t th{{background:var(--card2);color:var(--muted);text-transform:uppercase;
  letter-spacing:.04em;font-weight:600;padding:.5rem .7rem;text-align:right;
  border-bottom:1px solid var(--border);white-space:nowrap;cursor:pointer;user-select:none}}
table.t th:hover{{color:var(--accent)}}
table.t th.l,table.t td.l{{text-align:left}}
table.t td{{padding:.45rem .7rem;border-bottom:1px solid var(--card2);
  text-align:right;white-space:nowrap;color:var(--text)}}
table.t tr:hover td{{background:var(--card2)}}
table.t tr:last-child td{{border-bottom:none}}
table.t tr.row-profit td{{background:rgba(63,185,80,.06)!important}}
table.t tr.row-loss td{{background:rgba(248,81,73,.06)!important}}
table.t tr.row-neutral td{{background:transparent!important}}
table.t tr.row-profit:hover td{{background:rgba(63,185,80,.12)!important}}
table.t tr.row-loss:hover td{{background:rgba(248,81,73,.12)!important}}
.pos{{color:var(--green);font-weight:600}} .neg{{color:var(--red);font-weight:600}}
.profit-cell{{font-weight:700}}
.zero-cell{{color:var(--muted)!important;font-style:italic}}
.badge{{display:inline-block;padding:.12rem .4rem;border-radius:3px;
  font-size:.6rem;font-weight:700;text-transform:uppercase;letter-spacing:.04em}}
.b-open{{background:rgba(210,153,34,.18);color:var(--yellow)}}
.b-cl{{background:rgba(63,185,80,.15);color:var(--green)}}
.b-cll{{background:rgba(248,81,73,.15);color:var(--red)}}
.nse-lbl{{color:var(--muted);font-size:.68rem}}

.sig-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:.6rem;margin-top:.5rem}}
.sig-card{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:.75rem 1rem}}
.sig-card.sell{{border-left:3px solid var(--red)}}
.sig-card.avg{{border-left:3px solid var(--yellow)}}
.sig-card.hold{{border-left:3px solid var(--green)}}
.sig-card.watch{{border-left:3px solid var(--muted)}}
.sig-action{{font-size:.85rem;font-weight:700;margin-bottom:.25rem}}
.sig-meta{{font-size:.72rem;color:var(--muted);line-height:1.5}}
.sig-reason{{font-size:.72rem;margin-top:.3rem;color:var(--text);line-height:1.5}}
.sig-price{{font-size:.7rem;margin-top:.35rem;padding-top:.35rem;border-top:1px solid var(--border);line-height:1.6}}
.str-bar{{height:4px;border-radius:2px;margin-top:.4rem;background:var(--input)}}
.str-fill{{height:4px;border-radius:2px}}

.sector-tbl{{width:100%;border-collapse:collapse;font-size:.77rem}}
.sector-tbl th{{background:var(--card2);color:var(--muted);text-transform:uppercase;
  font-size:.65rem;letter-spacing:.05em;padding:.45rem .7rem;border-bottom:1px solid var(--border)}}
.sector-tbl td{{padding:.45rem .7rem;border-bottom:1px solid var(--card2);color:var(--text)}}
.sector-tbl tr:last-child td{{border-bottom:none}}
.sector-tbl tr:hover td{{background:var(--card2)}}

.pick-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:.6rem;margin-top:.5rem}}
.pick-card{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:.75rem 1rem;
  border-left:3px solid var(--accent)}}
.pick-card .pick-stock{{font-size:.9rem;font-weight:700}}
.pick-card .pick-sector{{font-size:.68rem;color:var(--muted)}}
.pick-card .pick-prices{{font-size:.72rem;margin-top:.4rem;line-height:1.7}}
.pick-card .pick-reason{{font-size:.68rem;color:var(--muted);margin-top:.35rem;
  border-top:1px solid var(--border);padding-top:.35rem}}

.outlook-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:.5rem;margin-top:.5rem}}
.outlook-card{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:.6rem .8rem}}
.outlook-card .outlook-sector{{font-size:.78rem;font-weight:700}}
.outlook-card .outlook-label{{font-size:.72rem;margin-top:.25rem}}
.outlook-card .outlook-meta{{font-size:.65rem;color:var(--muted);margin-top:.2rem}}

.stButton>button{{background:var(--input);border:1px solid var(--border);
  color:var(--text);border-radius:6px;font-size:.78rem;font-weight:500;
  padding:.35rem .7rem;transition:all .15s}}
.stButton>button:hover{{border-color:var(--accent);color:var(--accent);
  background:rgba(88,166,255,.08)}}

[data-testid="stSidebar"]{{background:var(--card);border-right:1px solid var(--border)}}
.stTextInput input,.stNumberInput input{{
  background:var(--input)!important;border:1px solid var(--border)!important;
  color:var(--text)!important;border-radius:6px!important;font-size:.8rem!important}}
.stSelectbox div[data-baseweb="select"]{{background:var(--input)!important;
  border:1px solid var(--border)!important;border-radius:6px!important}}
.stTabs [data-baseweb="tab-list"]{{background:var(--card);border-bottom:1px solid var(--border);gap:0;padding:0 .5rem}}
.stTabs [data-baseweb="tab"]{{background:transparent;color:var(--muted);font-size:.78rem;
  font-weight:500;padding:.5rem 1rem;border:none;border-bottom:2px solid transparent}}
.stTabs [aria-selected="true"]{{background:transparent!important;color:var(--accent)!important;
  border-bottom-color:var(--accent)!important}}
.js-plotly-plot .plotly{{border-radius:8px}}
.refresh-badge{{display:inline-block;background:rgba(63,185,80,.12);
  color:var(--green);padding:.15rem .5rem;border-radius:4px;
  font-size:.65rem;font-weight:600;margin-left:.5rem}}
.regime-banner{{border-radius:6px;padding:.5rem .85rem;display:flex;
  align-items:center;gap:.5rem;margin-bottom:.75rem;flex-wrap:wrap}}
  
[data-testid="stExpander"] {{
    background-color: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px;
    margin-bottom: 0.5rem;
}}
[data-testid="stExpander"] summary p {{
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    color: var(--text) !important;
}}
</style>
"""

# ── Database ───────────────────────────────────────────────────────────────────
DB = "trades.db"

def init_db():
    c = sqlite3.connect(DB)
    c.execute("""CREATE TABLE IF NOT EXISTS trades(
        id INTEGER PRIMARY KEY AUTOINCREMENT, stock TEXT NOT NULL,
        quantity REAL NOT NULL, buy_at REAL NOT NULL, sell_at REAL,
        status TEXT DEFAULT 'Open', added_date TEXT DEFAULT(date('now')),
        closed_date TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS portfolio_history(
        id INTEGER PRIMARY KEY AUTOINCREMENT, snapshot_date TEXT,
        total_invested REAL, current_value REAL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS tg_config(
        id INTEGER PRIMARY KEY, bot_token TEXT, chat_id TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS watchlist(
        id INTEGER PRIMARY KEY AUTOINCREMENT, stock TEXT NOT NULL,
        target_price REAL, notes TEXT, added_date TEXT DEFAULT(date('now')))""")
    c.execute("""CREATE TABLE IF NOT EXISTS price_alerts(
        id INTEGER PRIMARY KEY AUTOINCREMENT, stock TEXT NOT NULL,
        alert_type TEXT NOT NULL, price REAL NOT NULL,
        triggered INTEGER DEFAULT 0, created_date TEXT DEFAULT(date('now')))""")
    c.commit(); c.close()

def db(sql, params=(), fetch=False):
    conn = sqlite3.connect(DB)
    cur = conn.execute(sql, params)
    conn.commit()
    result = cur.fetchall() if fetch else None
    conn.close()
    return result

def get_trades():
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT * FROM trades ORDER BY id DESC", conn)
    conn.close()
    return df

def get_history():
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT * FROM portfolio_history ORDER BY snapshot_date", conn)
    conn.close()
    return df

def get_tg_config():
    rows = db("SELECT bot_token,chat_id FROM tg_config WHERE id=1", fetch=True)
    return rows[0] if rows else ("", "")

def save_tg_config(token, chat):
    db("INSERT OR REPLACE INTO tg_config(id,bot_token,chat_id) VALUES(1,?,?)", (token, chat))

def add_trade(stock, qty, buy, sell=None):
    status = "Closed" if sell else "Open"
    closed = datetime.now().strftime("%Y-%m-%d") if sell else None
    db("INSERT INTO trades(stock,quantity,buy_at,sell_at,status,closed_date) VALUES(?,?,?,?,?,?)",
       (stock.upper().strip(), qty, buy, sell, status, closed))

def update_trade(tid, stock, qty, buy, sell, status):
    closed = datetime.now().strftime("%Y-%m-%d") if status == "Closed" else None
    db("UPDATE trades SET stock=?,quantity=?,buy_at=?,sell_at=?,status=?,closed_date=? WHERE id=?",
       (stock.upper().strip(), qty, buy, sell, status, closed, tid))

def delete_trade(tid):
    db("DELETE FROM trades WHERE id=?", (tid,))

def close_trade(tid, sell):
    db("UPDATE trades SET sell_at=?,status='Closed',closed_date=? WHERE id=?",
       (sell, datetime.now().strftime("%Y-%m-%d"), tid))

def save_snapshot(invested, value):
    today = datetime.now().strftime("%Y-%m-%d")
    db("DELETE FROM portfolio_history WHERE snapshot_date=?", (today,))
    db("INSERT INTO portfolio_history(snapshot_date,total_invested,current_value) VALUES(?,?,?)",
       (today, invested, value))

def add_watchlist(stock, target=None, notes=""):
    db("INSERT INTO watchlist(stock,target_price,notes) VALUES(?,?,?)",
       (stock.upper().strip(), target, notes))

def get_watchlist():
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT * FROM watchlist ORDER BY id DESC", conn)
    conn.close()
    return df

def delete_watchlist_item(wid):
    db("DELETE FROM watchlist WHERE id=?", (wid,))

def add_alert(stock, alert_type, price):
    db("INSERT INTO price_alerts(stock,alert_type,price) VALUES(?,?,?)",
       (stock.upper().strip(), alert_type, price))

def get_alerts():
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT * FROM price_alerts WHERE triggered=0 ORDER BY id DESC", conn)
    conn.close()
    return df

def delete_alert(aid):
    db("DELETE FROM price_alerts WHERE id=?", (aid,))

def check_alerts(prices):
    alerts = get_alerts()
    triggered = []
    for _, a in alerts.iterrows():
        cmp = prices.get(a["stock"])
        if cmp is None:
            continue
        if a["alert_type"] == "above" and cmp >= a["price"]:
            triggered.append(f"🚨 {a['stock']} crossed ABOVE ₹{a['price']} — now ₹{cmp}")
            db("UPDATE price_alerts SET triggered=1 WHERE id=?", (a["id"],))
        elif a["alert_type"] == "below" and cmp <= a["price"]:
            triggered.append(f"🚨 {a['stock']} fell BELOW ₹{a['price']} — now ₹{cmp}")
            db("UPDATE price_alerts SET triggered=1 WHERE id=?", (a["id"],))
    return triggered

# ── Fast Price cache ──────────────────────────────────────────────────────────
_CACHE = {}
_TTL = 300

def fetch_price(symbol):
    key = symbol.upper()
    if key in _CACHE and time.time() - _CACHE[key][1] < _TTL:
        return _CACHE[key][0]

    for sfx in [".NS", ".BO"]:
        try:
            t = yf.Ticker(key + sfx)
            val = t.fast_info.get("last_price")
            if val is not None and not pd.isna(val):
                p = round(float(val), 2)
                _CACHE[key] = (p, time.time())
                return p
                
            h = t.history(period="1d", interval="1d", auto_adjust=True)
            if h is not None and not h.empty and "Close" in h.columns:
                p = round(float(h["Close"].iloc[-1]), 2)
                _CACHE[key] = (p, time.time())
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
    df["current_amt"] = np.where(df["status"] == "Open",
                                 df["quantity"] * df["cmp"].fillna(df["buy_at"]),
                                 df["quantity"] * df["sell_at"].fillna(df["buy_at"]))
    df["total_amt"] = np.where(df["sell_at"].notna(),
                               df["quantity"] * df["sell_at"], df["current_amt"])
    df["profit"] = df["total_amt"] - df["invested"]
    df["profit_pct"] = (df["profit"] / df["invested"] * 100).round(2)
    return df

# ── Trade Analytics ────────────────────────────────────────────────────────────
def calc_analytics(df):
    if df.empty:
        return {}
    closed = df[df["status"] == "Closed"]
    open_trades = df[df["status"] == "Open"]
    total = len(df)
    wins = len(closed[closed["profit"] > 0]) if not closed.empty else 0
    losses = len(closed[closed["profit"] < 0]) if not closed.empty else 0
    total_closed = len(closed)
    win_rate = (wins / total_closed * 100) if total_closed > 0 else 0
    avg_win = closed[closed["profit"] > 0]["profit"].mean() if wins > 0 else 0
    avg_loss = abs(closed[closed["profit"] < 0]["profit"].mean()) if losses > 0 else 0
    loss_rate = (losses / total_closed) if total_closed > 0 else 0
    win_rate_dec = (wins / total_closed) if total_closed > 0 else 0
    expectancy = (win_rate_dec * avg_win) - (loss_rate * avg_loss)
    gross_profit = closed[closed["profit"] > 0]["profit"].sum() if wins > 0 else 0
    gross_loss = abs(closed[closed["profit"] < 0]["profit"].sum()) if losses > 0 else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    if not closed.empty:
        cum_pnl = closed["profit"].cumsum()
        peak = cum_pnl.expanding().max()
        dd = cum_pnl - peak
        max_dd = abs(dd.min())
    else:
        max_dd = 0
    if not closed.empty and "closed_date" in closed.columns and "added_date" in closed.columns:
        closed_copy = closed.copy()
        closed_copy["hold_days"] = (
            pd.to_datetime(closed_copy["closed_date"]) -
            pd.to_datetime(closed_copy["added_date"])
        ).dt.days
        avg_hold = round(closed_copy["hold_days"].mean(), 1)
    else:
        avg_hold = 0
    if not closed.empty and closed["profit_pct"].std() > 0:
        sharpe = closed["profit_pct"].mean() / closed["profit_pct"].std()
    else:
        sharpe = 0
    return {
        "total_trades": total, "closed_trades": total_closed,
        "open_trades": len(open_trades), "wins": wins, "losses": losses,
        "win_rate": round(win_rate, 1), "avg_win": round(avg_win, 0),
        "avg_loss": round(avg_loss, 0), "expectancy": round(expectancy, 0),
        "profit_factor": round(profit_factor, 2), "max_drawdown": round(max_dd, 0),
        "avg_hold_days": avg_hold, "sharpe": round(sharpe, 2),
        "gross_profit": round(gross_profit, 0), "gross_loss": round(gross_loss, 0),
    }

# ── Formatting ─────────────────────────────────────────────────────────────────
def fi(v):
    return f"₹{v:,.0f}" if not pd.isna(v) else "—"

def fi2(v):
    return f"₹{v:,.2f}" if not pd.isna(v) else "—"

def fp(v):
    return f"{'+' if v >= 0 else ''}{v:.2f}%" if not pd.isna(v) else "—"

def cv(v, fn):
    s = fn(v)
    if pd.isna(v):
        return s
    c = "pos" if v > 0 else ("neg" if v < 0 else "zero-cell")
    return f'<span class="{c}">{s}</span>'

def cv_cell(v, fn):
    s = fn(v)
    if pd.isna(v):
        return f'<td>{s}</td>'
    if v > 0:
        return f'<td class="profit-cell pos">{s}</td>'
    elif v < 0:
        return f'<td class="profit-cell neg">{s}</td>'
    else:
        return f'<td class="zero-cell">{s}</td>'

def badge(status, profit=None):
    if status == "Open":
        return '<span class="badge b-open">Open</span>'
    return ('<span class="badge b-cll">Closed ✗</span>'
            if profit is not None and profit < 0
            else '<span class="badge b-cl">Closed ✓</span>')

def card(lbl, val, sub="", cls=""):
    return (f'<div class="card"><div class="lbl">{lbl}</div>'
            f'<div class="val {cls}">{val}</div>'
            f'{"<div class=sub>" + sub + "</div>" if sub else ""}</div>')

# ── Chart helpers ──────────────────────────────────────────────────────────────
CB = "#161b22"; CG = "#21262d"; CT = "#8b949e"

def base_layout(fig, title, theme_t=None):
    cb = theme_t.get("card", CB) if theme_t else CB
    fig.update_layout(
        title=dict(text=title, font=dict(size=12, color="#e6edf3"), x=.01, xanchor="left"),
        paper_bgcolor=cb, plot_bgcolor=cb, font=dict(color=CT, size=10),
        margin=dict(l=8, r=8, t=38, b=8),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=CT, size=9)))
    fig.update_xaxes(gridcolor=CG, zerolinecolor=CG, tickfont=dict(color=CT))
    fig.update_yaxes(gridcolor=CG, zerolinecolor=CG, tickfont=dict(color=CT))
    return fig

def chart_alloc(df, theme_t=None):
    d = df.groupby("stock")["invested"].sum().reset_index()
    fig = go.Figure(go.Pie(
        labels=d["stock"], values=d["invested"], hole=0,
        marker=dict(colors=px.colors.qualitative.Dark24, line=dict(color="#0d1117", width=2)),
        textinfo="percent+label", textfont=dict(size=10, color="#e6edf3")))
    return base_layout(fig, "Portfolio Allocation", theme_t)

def chart_pnl(df, theme_t=None):
    d = df.sort_values("profit")
    cols = ["#f85149" if v < 0 else "#3fb950" for v in d["profit"]]
    cb = theme_t.get("card", CB) if theme_t else CB
    fig = go.Figure(go.Bar(
        x=d["profit"], y=d["stock"], orientation="h",
        marker=dict(color=cols, line=dict(width=0)),
        text=[fp(p) for p in d["profit_pct"]],
        textposition="outside", textfont=dict(color="#e6edf3", size=9),
        hovertemplate="<b>%{y}</b><br>₹%{x:,.0f}<extra></extra>"))
    fig.update_layout(paper_bgcolor=cb, plot_bgcolor=cb, font=dict(color=CT, size=10),
                      margin=dict(l=8, r=45, t=38, b=8), showlegend=False,
                      title=dict(text="P&L by Stock", font=dict(size=12, color="#e6edf3"), x=.01, xanchor="left"),
                      xaxis=dict(gridcolor=CG, zerolinecolor="#388bfd", tickprefix="₹", tickfont=dict(color=CT)),
                      yaxis=dict(gridcolor=CG, tickfont=dict(color=CT)))
    return fig

def chart_donut(df, theme_t=None):
    counts = df["status"].value_counts().reset_index()
    counts.columns = ["Status", "Count"]
    cols = {"Open": "#d29922", "Closed": "#3fb950"}
    fig = go.Figure(go.Pie(
        labels=counts["Status"], values=counts["Count"], hole=.55,
        marker=dict(colors=[cols.get(s, "#8b949e") for s in counts["Status"]],
                    line=dict(color="#0d1117", width=3)),
        textinfo="percent+value", textfont=dict(size=11, color="#e6edf3")))
    fig.add_annotation(text=f"{len(df)}<br><span style='font-size:8px'>TRADES</span>",
                       font=dict(size=15, color="#e6edf3"), showarrow=False, x=.5, y=.5)
    return base_layout(fig, "Open vs Closed", theme_t)

def chart_growth(hist, cur_val, cur_inv, theme_t=None):
    today = datetime.now().strftime("%Y-%m-%d")
    rows = hist[["snapshot_date", "total_invested", "current_value"]].to_dict("records") if not hist.empty else []
    if not rows or rows[-1]["snapshot_date"] != today:
        rows.append({"snapshot_date": today, "total_invested": cur_inv, "current_value": cur_val})
    d = pd.DataFrame(rows)
    d["snapshot_date"] = pd.to_datetime(d["snapshot_date"])
    cb = theme_t.get("card", CB) if theme_t else CB
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["snapshot_date"], y=d["current_value"], name="Value",
                             line=dict(color="#3fb950", width=2), fill="tozeroy", fillcolor="rgba(63,185,80,.07)",
                             hovertemplate="%{x|%d %b}<br>₹%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=d["snapshot_date"], y=d["total_invested"], name="Invested",
                             line=dict(color="#388bfd", width=1.5, dash="dot"),
                             hovertemplate="%{x|%d %b}<br>₹%{y:,.0f}<extra></extra>"))
    fig.update_layout(paper_bgcolor=cb, plot_bgcolor=cb, font=dict(color=CT, size=10),
                      margin=dict(l=8, r=8, t=38, b=8), hovermode="x unified",
                      title=dict(text="Portfolio Growth", font=dict(size=12, color="#e6edf3"), x=.01, xanchor="left"),
                      xaxis=dict(gridcolor=CG, zerolinecolor=CG, tickfont=dict(color=CT)),
                      yaxis=dict(gridcolor=CG, zerolinecolor=CG, tickfont=dict(color=CT), tickprefix="₹"),
                      legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=CT, size=9)))
    return fig

# ── Render helpers ─────────────────────────────────────────────────────────────
def render_signals(signals, theme_t):
    if not signals:
        st.info("No signals available.")
        return
    t = theme_t
    html = '<div class="sig-grid">'
    for s in signals:
        act = s.get("action", "")
        cls = ("sell" if "SELL" in act else "avg" if "AVERAGE" in act
               else "hold" if "HOLD" in act else "watch")
        clr = (t["red"] if cls == "sell" else t["yellow"] if cls == "avg"
               else t["green"] if cls == "hold" else t["muted"])
        width = s.get("strength", 30)
        
        # ── SAFE STRING EXTRACTORS (Prevents "None" from breaking UI) ──
        cmp_s = f"₹{s['cmp']}" if s.get("cmp") is not None else "—"
        rsi_s = f"{s['rsi']:.2f}" if s.get("rsi") is not None else "—"
        pct_s = f"{s.get('pct_from_buy', 0):+.1f}%" if s.get("pct_from_buy") is not None else "—"
        trend = s.get("trend", "—")
        macd = s.get("macd_signal", "—")
        bb = s.get("bb_position", "—")
        rr = s.get("risk_reward") if s.get("risk_reward") is not None else "—"
        div = s.get("divergence", "—")
        st_lbl = s.get("supertrend", "—")
        regime = s.get("market_regime", "—")

        tgt_s = f"₹{s['target']}" if s.get("target") is not None else "—"
        sl_s = f"₹{s['stop_loss']}" if s.get("stop_loss") is not None else "—"
        avg_s = f"₹{s['avg_price']}" if s.get("avg_price") is not None else "—"
        navg_s = f"₹{s['new_avg']}" if s.get("new_avg") is not None else "—"
        nsl_s = f"₹{s['new_sl']}" if s.get("new_sl") is not None else "—"

        price_html = ""
        if cls == "sell":
            price_html = (f'<div class="sig-price">'
                          f'🎯 <b>Exit:</b> {tgt_s} | 🛑 <b>Re-entry:</b> {sl_s}<br>'
                          f'📉 {trend} | MACD: {macd} | Div: {div}<br>'
                          f'🌐 Market: {regime} | ST: {st_lbl}</div>')
        elif cls == "avg":
            price_html = (f'<div class="sig-price">'
                          f'💰 <b>Avg at:</b> {avg_s} | <b>New Avg:</b> {navg_s}<br>'
                          f'🛑 <b>New SL:</b> {nsl_s} | 🎯 <b>Target:</b> {tgt_s}<br>'
                          f'📊 R:R {rr} | {trend} | MACD: {macd} | Div: {div}</div>')
        elif cls == "hold":
            price_html = (f'<div class="sig-price">'
                          f'🎯 <b>Target:</b> {tgt_s} | 🛑 <b>SL:</b> {sl_s}<br>'
                          f'📊 R:R {rr} | {trend} | MACD: {macd} | BB: {bb}</div>')
        else:
            price_html = (f'<div class="sig-price">'
                          f'🎯 {tgt_s} | 🛑 {sl_s}<br>'
                          f'📊 {trend} | MACD: {macd} | Div: {div}</div>')

        html += f"""
        <div class="sig-card {cls}">
          <div class="sig-action" style="color:{clr}">{act}</div>
          <div style="font-size:.82rem;font-weight:700;margin-bottom:.2rem">{s['stock']}
            <span class="nse-lbl">{s.get('sector', '')}</span></div>
          <div class="sig-meta">CMP {cmp_s} · RSI {rsi_s} · From Buy {pct_s}</div>
          <div class="sig-reason">{s.get('reason', '')}</div>
          {price_html}
          <div class="str-bar"><div class="str-fill" style="width:{width}%;background:{clr}"></div></div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def render_sector(sector_df, theme_t):
    if sector_df is None or sector_df.empty:
        st.info("No sector data loaded.")
        return
    t = theme_t
    rows = ""
    for _, r in sector_df.iterrows():
        emoji = "🥇" if r["rank"] == 1 else ("🥈" if r["rank"] == 2 else ("🥉" if r["rank"] == 3 else "📊"))
        score_bar = (f'<div style="background:{t["input"]};border-radius:3px;height:5px;width:100%;margin-top:3px">'
                     f'<div style="background:{t["accent"]};width:{min(r["momentum_score"]*100,100):.0f}%;height:5px;border-radius:3px"></div></div>')
        pct_col = f'<span class="pos">{r["avg_pct"]:+.1f}%</span>' if r["avg_pct"] > 0 else f'<span class="neg">{r["avg_pct"]:+.1f}%</span>'
        bullish = int(r.get("bullish_count", 0))
        bull_lbl = f'<span style="color:{t["green"]};font-size:.68rem">🐂 ×{bullish}</span>' if bullish else ""
        idx_chg = f'<span style="font-size:.65rem;color:var(--muted)">Idx:{r["index_chg"]:+.1f}%</span>' if pd.notna(r.get("index_chg")) else ""
        rows += f"""<tr>
          <td>{emoji} #{int(r['rank'])}</td>
          <td><b>{r['sector']}</b></td>
          <td style="color:var(--muted);font-size:.7rem">{r['stocks']}</td>
          <td style="text-align:center">{r['avg_rsi']:.0f}</td>
          <td style="text-align:right">{pct_col} {idx_chg}</td>
          <td>{score_bar}<span style="font-size:.68rem;color:var(--muted)">{r['momentum_score']:.2f}</span> {bull_lbl}</td>
        </tr>"""
    st.markdown(f"""
    <div style="overflow-x:auto;background:var(--card);border:1px solid var(--border);border-radius:8px">
    <table class="sector-tbl">
      <thead><tr><th>Rank</th><th>Sector</th><th>Stocks</th><th style="text-align:center">Avg RSI</th>
        <th style="text-align:right">Avg Chg</th><th>Momentum</th></tr></thead>
      <tbody>{rows}</tbody>
    </table></div>""", unsafe_allow_html=True)

def render_outlook(outlook_df, theme_t):
    if outlook_df is None or outlook_df.empty:
        return
    html = '<div class="outlook-grid">'
    for _, r in outlook_df.iterrows():
        html += f"""
        <div class="outlook-card">
          <div class="outlook-sector">{r['sector']}</div>
          <div class="outlook-label">{r['outlook']}</div>
          <div class="outlook-meta">
            Confidence: {r['confidence']}% · Momentum: {r['momentum']:.2f}<br>
            Avg RSI: {r['avg_rsi']:.0f} · Avg Chg: {r['avg_pct']:+.1f}%
          </div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def render_picks(picks, theme_t):
    if not picks:
        st.info("No buy picks found.")
        return
    t = theme_t
    html = '<div class="pick-grid">'
    for p in picks:
        sc = t["green"] if p["score"] >= 70 else (t["yellow"] if p["score"] >= 55 else t["muted"])
        html += f"""
        <div class="pick-card" style="border-left-color:{sc}">
          <div class="pick-stock">{p['stock']} <span class="pick-sector">{p['sector']}</span></div>
          <div style="font-size:.75rem;color:{t['muted']}">CMP ₹{p['cmp']} · RSI {p['rsi']} · {p['trend']}</div>
          <div class="pick-prices">
            🎯 <b>Entry:</b> ₹{p['entry']}<br>
            🚀 <b>Target:</b> ₹{p['target']}<br>
            🛑 <b>Stop Loss:</b> ₹{p['stop_loss']}<br>
            📊 <b>R:R:</b> {p['risk_reward']} · Score: {p['score']}
          </div>
          <div class="pick-reason">{p['reason']}</div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# ── Init & Session State ──────────────────────────────────────────────────────
init_db()
for k, v in [("edit_id", None), ("close_id", None), ("del_id", None),
             ("last_refresh", None), ("last_auto_scan", 0.0), ("sort_col", "stock"), ("sort_asc", False),
             ("signals_cache", None), ("sector_cache", None), ("picks_cache", None),
             ("outlook_cache", None), ("scanner_cache", None), ("filter_status", "All"),
             ("filter_pnl", "All"), ("search", ""), ("theme", "Midnight"),
             ("alerts_triggered", [])]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Load data ──────────────────────────────────────────────────────────────────
raw = get_trades()
df = enrich(raw) if not raw.empty else raw.copy()

if st.session_state.last_refresh is None or \
        (datetime.now() - st.session_state.last_refresh).seconds >= _TTL:
    st.session_state.last_refresh = datetime.now()

# ── 🤖 CRITICAL FIX: 15-MINUTE TIMED BACKGROUND AUTO SCAN ENGINE ──────────────
SCAN_INTERVAL_SEC = 900 # 15 Minutes
current_timestamp = time.time()
time_elapsed = current_timestamp - st.session_state.last_auto_scan

if st.session_state.last_auto_scan == 0.0 or time_elapsed >= SCAN_INTERVAL_SEC:
    with st.spinner("🤖 Running Scheduled 15-Min Background Market Scan..."):
        # 1. Scan Signals
        open_df = df[df["status"] == "Open"] if not df.empty else pd.DataFrame()
        if not open_df.empty:
            st.session_state.signals_cache = generate_signals(raw[raw["status"] == "Open"])
        else:
            st.session_state.signals_cache = []
            
        # 2. Scan Sectors
        st.session_state.sector_cache = sector_rotation()
        if st.session_state.sector_cache is not None and not st.session_state.sector_cache.empty:
            st.session_state.outlook_cache = predict_sector_outlook(st.session_state.sector_cache)
            top_sectors = st.session_state.sector_cache.head(5)["sector"].tolist()
            st.session_state.picks_cache = find_sector_picks(top_sectors, max_per_sector=3)
        else:
            st.session_state.outlook_cache = pd.DataFrame()
            st.session_state.picks_cache = []
            
        # 3. Scan Global F&O Universe
        st.session_state.scanner_cache = generate_market_scanner()
        
        # Log successful completion timestamp
        st.session_state.last_auto_scan = current_timestamp

# Summary numbers
if not df.empty:
    open_df = df[df["status"] == "Open"]
    closed_df = df[df["status"] == "Closed"]
    t_inv = df["invested"].sum()
    t_cur = df["current_amt"].sum()
    t_real = closed_df["profit"].sum() if not closed_df.empty else 0
    t_unreal = open_df["profit"].sum() if not open_df.empty else 0
    t_pnl = df["profit"].sum()
    t_pnl_pct = t_pnl / t_inv * 100 if t_inv > 0 else 0
    best = df.loc[df["profit_pct"].idxmax(), "stock"]
    worst = df.loc[df["profit_pct"].idxmin(), "stock"]
    save_snapshot(t_inv, t_cur)
else:
    open_df = closed_df = pd.DataFrame()
    t_inv = t_cur = t_real = t_unreal = t_pnl = t_pnl_pct = 0
    best = worst = "—"

# ── Apply theme ────────────────────────────────────────────────────────────────
theme_t = THEMES[st.session_state.theme]
st.markdown(theme_css(theme_t), unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-size:.72rem;font-weight:600;color:var(--muted);margin-bottom:.4rem">🎨 THEME</div>',
                unsafe_allow_html=True)
    new_theme = st.selectbox("Theme", list(THEMES.keys()),
                             index=list(THEMES.keys()).index(st.session_state.theme),
                             label_visibility="collapsed")
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

    st.markdown("---")
    st.markdown('<div style="font-size:1rem;font-weight:700;color:#e6edf3;margin-bottom:.75rem">⚡ Trade Entry</div>',
                unsafe_allow_html=True)
    edit_mode = st.session_state.edit_id is not None
    erow = raw[raw["id"] == st.session_state.edit_id].iloc[0] if edit_mode and not raw.empty else None

    if edit_mode:
        st.markdown('<div style="background:rgba(88,166,255,.1);border:1px solid rgba(88,166,255,.3);'
                    'border-radius:5px;padding:.35rem .6rem;font-size:.72rem;color:#58a6ff;margin-bottom:.5rem">'
                    '✏️ Editing trade</div>', unsafe_allow_html=True)

    with st.form("trade_form", clear_on_submit=True):
        s_in = st.text_input("Stock Symbol", value=erow["stock"] if erow is not None else "",
                             placeholder="CDSL, IRFC…")
        q_in = st.number_input("Quantity", min_value=1, step=1,
                               value=int(erow["quantity"]) if erow is not None else 1)
        b_in = st.number_input("Buy At ₹", min_value=0.01, step=0.05,
                               value=float(erow["buy_at"]) if erow is not None else 0.01, format="%.2f")
        sel_in = st.number_input("Sell At ₹ (optional)", min_value=0.0, step=0.05,
                                 value=float(erow["sell_at"]) if (erow is not None and erow["sell_at"]) else 0.0,
                                 format="%.2f")
        ok = st.form_submit_button("💾 Update" if edit_mode else "➕ Add Trade", use_container_width=True)

    if ok:
        if not s_in.strip():
            st.error("Symbol required")
        elif b_in <= 0:
            st.error("Buy price must be > 0")
        else:
            sv = sel_in if sel_in > 0 else None
            if edit_mode:
                update_trade(st.session_state.edit_id, s_in, q_in, b_in, sv,
                             "Closed" if sv else "Open")
                st.session_state.edit_id = None
                st.success("Updated!")
            else:
                add_trade(s_in, q_in, b_in, sv)
                st.success(f"Added {s_in.upper()}")
            
            # Force a manual data clearing to prompt cache re-population
            _CACHE.clear()
            st.session_state.last_auto_scan = 0.0
            st.rerun()

    if edit_mode and st.button("✖ Cancel Edit", use_container_width=True):
        st.session_state.edit_id = None
        st.rerun()

    st.markdown("---")
    st.markdown('<div style="font-size:.72rem;font-weight:600;color:#8b949e;margin-bottom:.4rem">FILTERS</div>',
                unsafe_allow_html=True)
    st.session_state.filter_status = st.selectbox(
        "Status", ["All", "Open", "Closed"],
        index=["All", "Open", "Closed"].index(st.session_state.filter_status),
        label_visibility="collapsed")
    st.session_state.filter_pnl = st.selectbox(
        "P&L", ["All", "Profitable", "Loss"],
        index=["All", "Profitable", "Loss"].index(st.session_state.filter_pnl),
        label_visibility="collapsed")
    st.session_state.search = st.text_input(
        "Search", value=st.session_state.search, placeholder="Search symbol…",
        label_visibility="collapsed")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Force Scan", use_container_width=True):
            _CACHE.clear()
            st.session_state.last_refresh = datetime.now()
            st.session_state.last_auto_scan = 0.0  
            st.rerun()
    with c2:
        next_scan_min = max(0, int((SCAN_INTERVAL_SEC - (time.time() - st.session_state.last_auto_scan)) // 60))
        st.markdown(f'<div style="font-size:.62rem;color:#8b949e;padding-top:.3rem">Next: {next_scan_min}m</div>',
                    unsafe_allow_html=True)

    st.markdown(f'<div style="font-size:.6rem;color:var(--green);text-align:center">'
                f'⚡ UI Refresh: {REFRESH_SEC // 60}m | Auto Scan: 15m</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-size:.72rem;font-weight:600;color:#8b949e;margin-bottom:.4rem">TELEGRAM</div>',
                unsafe_allow_html=True)
    saved_tok, saved_cid = get_tg_config()
    tg_tok = st.text_input("Bot Token", value=saved_tok, type="password", placeholder="123456:ABC…")
    tg_cid = st.text_input("Chat ID", value=saved_cid, placeholder="-100xxxxxxx")
    if st.button("💾 Save Telegram", use_container_width=True):
        save_tg_config(tg_tok, tg_cid)
        st.success("Saved!")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f'<div class="dash-title">📈 Swing Trade <span>Dashboard</span> '
            f'<span class="refresh-badge">AUTO SCAN 15m</span></div>',
            unsafe_allow_html=True)

# ── Market Regime Banner ──────────────────────────────────────────────────────
market = get_market_regime()
regime_colors = {
    "Strong Bull": ("rgba(63,185,80,.12)", "#3fb950"),
    "Bull": ("rgba(63,185,80,.08)", "#3fb950"),
    "Bull Pullback": ("rgba(210,153,34,.1)", "#d29922"),
    "Strong Bear": ("rgba(248,81,73,.12)", "#f85149"),
    "Bear": ("rgba(248,81,73,.08)", "#f85149"),
    "Bear Rally": ("rgba(210,153,34,.1)", "#d29922"),
}
rc_bg, rc_clr = regime_colors.get(market["regime"], ("rgba(139,148,158,.08)", "#8b949e"))

indices_html = ""
for idx_name, idx_data in market.get("indices", {}).items():
    price = f"₹{idx_data['price']:,.0f}" if idx_data.get('price') else "—"
    chg = idx_data.get('chg_pct', 0)
    if idx_name == "India VIX":
        chg_color = "var(--red)" if chg > 0 else ("var(--green)" if chg < 0 else "var(--muted)")
    else:
        chg_color = "var(--green)" if chg > 0 else ("var(--red)" if chg < 0 else "var(--muted)")
    chg_str = f"{chg:+.2f}%"
    indices_html += (
        f'<span style="color:var(--text);font-size:.75rem;padding:0 .7rem;'
        f'border-right:1px solid var(--border)">'
        f'{idx_name} <b>{price}</b> '
        f'<span style="color:{chg_color}">{chg_str}</span></span>'
    )

rsi_lbl = f"RSI {market.get('nifty_rsi', '—')}" 
sup_val = market.get("support")
res_val = market.get("resistance")
sup_str = f"₹{sup_val:,.0f}" if sup_val else "—"
res_str = f"₹{res_val:,.0f}" if res_val else "—"
conf_str = market.get('confidence', '—')

st.markdown(
    f'<div class="regime-banner" style="background:{rc_bg};border:1px solid {rc_clr}">'
    f'<span style="color:{rc_clr};font-weight:700;font-size:.85rem;white-space:nowrap">🌐 {market["regime"]} (Conf: {conf_str}%)</span>'
    f'{indices_html}'
    f'<span style="color:var(--muted);font-size:.7rem;white-space:nowrap;padding-left:0.5rem;">Sup: {sup_str} | Res: {res_str} | {rsi_lbl} | Risk: {market.get("risk_level","—")}</span>'
    f'</div>', unsafe_allow_html=True)

# ── Summary cards ──────────────────────────────────────────────────────────────
pnl_c = "green" if t_pnl >= 0 else "red"
r_c = "green" if t_real >= 0 else "red"
u_c = "green" if t_unreal >= 0 else "red"

st.markdown(
    '<div class="cards">'
    + card("Total Invested", fi(t_inv), "", "blue")
    + card("Portfolio Value", fi(t_cur), "", "blue")
    + card("Total P&L", fi(t_pnl), fp(t_pnl_pct), pnl_c)
    + card("Realized P&L", fi(t_real), "", r_c)
    + card("Unrealized P&L", fi(t_unreal), "", u_c)
    + card("Open", str(len(open_df)), "trades", "yellow")
    + card("Closed", str(len(closed_df)), "trades", "green" if len(closed_df) > 0 else "")
    + card("Best 🏆", best, "", "green")
    + card("Worst 📉", worst, "", "red")
    + '</div>', unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    ["📋 Trades", "📊 Charts", "🔔 Signals", "🔄 Sectors", "🌌 Universe Scan",
     "📐 Analytics", "👁 Watchlist", "📤 Export"]
)

with tab1:
    if df.empty:
        st.info("No trades yet. Add one using the sidebar.")
    else:
        fdf = df.copy()
        if st.session_state.filter_status != "All":
            fdf = fdf[fdf["status"] == st.session_state.filter_status]
        if st.session_state.filter_pnl == "Profitable":
            fdf = fdf[fdf["profit"] > 0]
        elif st.session_state.filter_pnl == "Loss":
            fdf = fdf[fdf["profit"] < 0]
        if st.session_state.search.strip():
            fdf = fdf[fdf["stock"].str.upper().str.contains(st.session_state.search.upper())]

        sort_opts = {
            "Stock": "stock", "Qty": "quantity", "Buy At": "buy_at",
            "CMP": "cmp", "Invested": "invested", "P&L ₹": "profit", "P&L %": "profit_pct"
        }
        valid_sort_vals = list(sort_opts.values())
        default_col = st.session_state.sort_col if st.session_state.sort_col in valid_sort_vals else "stock"
        default_idx = valid_sort_vals.index(default_col)

        sc1, sc2 = st.columns([3, 1])
        with sc1:
            sort_label = st.selectbox("Sort by", list(sort_opts.keys()),
                                      index=default_idx, label_visibility="collapsed")
        with sc2:
            asc = st.toggle("⬆", value=st.session_state.sort_asc)
            st.session_state.sort_asc = asc

        sort_col = sort_opts[sort_label]
        st.session_state.sort_col = sort_col
        if sort_col in fdf.columns:
            fdf = fdf.sort_values(sort_col, ascending=asc, na_position="last")

        st.markdown(f'<div class="sec">{len(fdf)} trade(s)</div>', unsafe_allow_html=True)

        rows_html = ""
        for _, r in fdf.iterrows():
            sa = fi2(r["sell_at"]) if pd.notna(r.get("sell_at")) else "—"
            pv = r.get("profit", 0)
            pp = r.get("profit_pct", 0)
            sector_lbl = get_sector(r["stock"])
            status = r["status"]
            row_cls = "row-profit" if pv > 0 else ("row-loss" if pv < 0 else "row-neutral")

            cmp_val = r.get("cmp")
            if pd.notna(cmp_val):
                cmp_html = f'<td class="pos">{fi2(cmp_val)}</td>' if cmp_val > r["buy_at"] \
                    else (f'<td class="neg">{fi2(cmp_val)}</td>' if cmp_val < r["buy_at"]
                          else f'<td>{fi2(cmp_val)}</td>')
            else:
                cmp_html = '<td class="zero-cell">—</td>'

            curr_amt = r.get("current_amt", 0)
            if pd.notna(curr_amt):
                curr_html = f'<td class="pos">{fi(curr_amt)}</td>' if curr_amt > r["invested"] \
                    else (f'<td class="neg">{fi(curr_amt)}</td>' if curr_amt < r["invested"]
                          else f'<td>{fi(curr_amt)}</td>')
            else:
                curr_html = '<td>—</td>'

            rows_html += f"""<tr class="{row_cls}">
              <td class="l"><span class="nse-lbl">{r.get('nse_label','')}</span></td>
              <td class="l"><b>{r['stock']}</b><br>
                <span class="nse-lbl">{sector_lbl} · {r.get('added_date','')}</span></td>
              <td>{int(r['quantity'])}</td>
              <td>{fi2(r['buy_at'])}</td>
              {cmp_html}
              <td>{sa}</td>
              <td>{fi(r['invested'])}</td>
              {curr_html}
              {cv_cell(pv, fi)}
              {cv_cell(pp, fp)}
              <td>{badge(status, pv)}</td>
            </tr>"""

        st.markdown(
            f'<div class="tbl-wrap"><table class="t">'
            f'<thead><tr>'
            f'<th class="l">NSE</th><th class="l">Stock</th><th>Qty</th>'
            f'<th>Buy At</th><th>CMP</th><th>Sell At</th>'
            f'<th>Invested</th><th>Curr Amt</th>'
            f'<th>Profit ₹</th><th>Profit %</th><th>Status</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>',
            unsafe_allow_html=True)

        st.markdown('<div class="sec">Trade Actions</div>', unsafe_allow_html=True)
        opts = [f"{r['id']} — {r['stock']}" for _, r in fdf.iterrows()]
        if opts:
            ca, cb, cc, cd = st.columns([3, 1, 1, 1])
            with ca:
                sel = st.selectbox("Trade", opts, label_visibility="collapsed")
                sel_id = int(sel.split(" — ")[0])
            with cb:
                if st.button("✏️ Edit", use_container_width=True):
                    st.session_state.edit_id = sel_id; st.rerun()
            with cc:
                if st.button("🔒 Close", use_container_width=True):
                    st.session_state.close_id = sel_id; st.rerun()
            with cd:
                if st.button("🗑 Delete", use_container_width=True):
                    st.session_state.del_id = sel_id; st.rerun()

        if st.session_state.close_id:
            st.markdown("---")
            st.markdown("**Close Trade — Enter Sell Price**")
            sp = st.number_input("Sell Price ₹", min_value=0.01, step=0.05, format="%.2f")
            x1, x2 = st.columns(2)
            with x1:
                if st.button("✅ Confirm Close", use_container_width=True):
                    close_trade(st.session_state.close_id, sp)
                    st.session_state.close_id = None; st.rerun()
            with x2:
                if st.button("✖ Cancel Close", use_container_width=True):
                    st.session_state.close_id = None; st.rerun()

        if st.session_state.del_id:
            st.markdown("---")
            st.warning(f"Delete trade #{st.session_state.del_id}? Cannot be undone.")
            y1, y2 = st.columns(2)
            with y1:
                if st.button("🗑 Confirm Delete", use_container_width=True):
                    delete_trade(st.session_state.del_id)
                    st.session_state.del_id = None; st.rerun()
            with y2:
                if st.button("✖ Keep Trade", use_container_width=True):
                    st.session_state.del_id = None; st.rerun()

with tab2:
    if df.empty:
        st.info("Add trades to see charts.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(chart_alloc(df, theme_t), use_container_width=True)
        with c2:
            st.plotly_chart(chart_donut(df, theme_t), use_container_width=True)
        st.plotly_chart(chart_pnl(df, theme_t), use_container_width=True)
        st.plotly_chart(chart_growth(get_history(), t_cur, t_inv, theme_t), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3 — SIGNALS (Automated Representation)
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec">Expert Trading Signals — Open Positions</div>', unsafe_allow_html=True)
    
    s1, s2 = st.columns([2, 1])
    with s1:
        st.caption("🔄 Scans continuously in the background every 15 minutes.")
    with s2:
        tg_enabled = bool(saved_tok and saved_cid)
        if st.button("📲 Send to Telegram", use_container_width=True, disabled=not tg_enabled):
            if st.session_state.signals_cache is not None:
                sec_df = st.session_state.sector_cache if st.session_state.sector_cache is not None else pd.DataFrame()
                picks = st.session_state.picks_cache if st.session_state.picks_cache is not None else []
                msg = build_telegram_message(st.session_state.signals_cache, sec_df, picks)
                ok = send_telegram(saved_tok, saved_cid, msg)
                st.success("✅ Sent!") if ok else st.error("❌ Failed.")
        if not tg_enabled:
            st.caption("Configure Telegram in sidebar.")

    if st.session_state.signals_cache is not None:
        sigs = st.session_state.signals_cache
        nc = {"SELL": 0, "AVERAGE": 0, "HOLD": 0, "WATCH": 0}
        for s in sigs:
            for k in nc:
                if k in s.get("action", ""):
                    nc[k] += 1
        st.markdown(
            f'<div style="display:flex;gap:.5rem;margin:.5rem 0 .75rem">'
            f'<span style="background:rgba(248,81,73,.15);color:#f85149;padding:.2rem .6rem;border-radius:4px;font-size:.72rem;font-weight:700">🔴 SELL: {nc["SELL"]}</span>'
            f'<span style="background:rgba(210,153,34,.15);color:#d29922;padding:.2rem .6rem;border-radius:4px;font-size:.72rem;font-weight:700">🟡 AVERAGE: {nc["AVERAGE"]}</span>'
            f'<span style="background:rgba(63,185,80,.15);color:#3fb950;padding:.2rem .6rem;border-radius:4px;font-size:.72rem;font-weight:700">🟢 HOLD: {nc["HOLD"]}</span>'
            f'<span style="background:rgba(139,148,158,.15);color:#8b949e;padding:.2rem .6rem;border-radius:4px;font-size:.72rem;font-weight:700">⚪ WATCH: {nc["WATCH"]}</span>'
            f'</div>', unsafe_allow_html=True)
        render_signals(sigs, theme_t)

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 4 — SECTOR ROTATION (Automated Representation)
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec">Sector Rotation — Momentum Ranking</div>', unsafe_allow_html=True)

    if st.session_state.sector_cache is not None:
        render_sector(st.session_state.sector_cache, theme_t)
        if not st.session_state.sector_cache.empty:
            top = st.session_state.sector_cache.iloc[0]
            st.markdown(
                f'<div style="margin-top:.75rem;background:rgba(63,185,80,.07);border:1px solid rgba(63,185,80,.2);'
                f'border-radius:6px;padding:.5rem .75rem;font-size:.78rem">'
                f'🥇 <b>Top sector: {top["sector"]}</b> — Momentum {top["momentum_score"]:.2f} | '
                f'Avg RSI {top["avg_rsi"]:.0f} | Avg change {top["avg_pct"]:+.1f}% — '
                f'<span style="color:#8b949e">Consider: {top["stocks"]}</span></div>',
                unsafe_allow_html=True)

    if st.session_state.outlook_cache is not None and not st.session_state.outlook_cache.empty:
        st.markdown('<div class="sec">📈 Sector Outlook — Next Week</div>', unsafe_allow_html=True)
        render_outlook(st.session_state.outlook_cache, theme_t)

    if st.session_state.picks_cache is not None:
        st.markdown('<div class="sec">🎯 Potential Buy Picks</div>', unsafe_allow_html=True)
        render_picks(st.session_state.picks_cache, theme_t)

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 5 — UNIVERSE SCANNER (God Mode)
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    total_loaded_stocks = len(SECTOR_MAP)
    
    st.markdown(f'<div class="sec">🌌 Complete Market Scan ({total_loaded_stocks} Stocks Loaded)</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:rgba(88,166,255,.07);border:1px solid rgba(88,166,255,.2);
    border-radius:6px;padding:.5rem .75rem;font-size:.73rem;color:#8b949e;margin-bottom:.75rem">
    Analyzes all <b>{total_loaded_stocks} stocks</b> independently from your portfolio.<br>
    Filters by Sector, calculates institutional confluence, and assigns a master strength signal.<br>
    <i>Note: This triggers bulk API calls. It will take ~15 to 25 seconds to complete.</i>
    </div>
    """, unsafe_allow_html=True)

    if st.button("⚡ Force Full Universe Scan Now", use_container_width=True):
        with st.spinner(f"Running deep background & indicator calculations across {total_loaded_stocks} tickers..."):
            try:
                scanned_data = generate_market_scanner()
                
                if scanned_data is not None and not scanned_data.empty:
                    st.session_state.scanner_cache = scanned_data
                    st.toast(f"✅ Successfully scanned and filtered {len(scanned_data)} liquid stocks!", icon="🚀")
                else:
                    st.session_state.scanner_cache = pd.DataFrame()
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Universe Scan Engine Failed: {str(e)}")
                st.info("Check your internet connection, Yahoo Finance rate status, or your CSV column headers.")
    
    scan_df = st.session_state.scanner_cache
    
    if scan_df is None:
        st.info("💡 Click the button above or wait for the 15-minute auto-scan interval to load market data.")
    elif scan_df.empty:
        st.warning("⚠️ Scanner returned 0 rows. Check if the market is closed, if your CSV symbols match yFinance, or if the liquidity gate is too strict.")
    else:
        sectors = scan_df["Sector"].unique()
        for sec in sectors:
            sec_df = scan_df[scan_df["Sector"] == sec].drop(columns=["Sector"])
            bullish_count = len(sec_df[sec_df["Score"] >= 4])
            badge_text = f"🔥 {bullish_count} Setups" if bullish_count > 0 else ""
            
            with st.expander(f"📁 {sec} ({len(sec_df)} stocks) {badge_text}"):
                st.dataframe(
                    sec_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Generated": st.column_config.TextColumn("Setup Time"),
                        "CMP": st.column_config.NumberColumn("CMP", format="₹%.2f"),
                        "Entry": st.column_config.NumberColumn("Entry", format="₹%.2f"),
                        "Target": st.column_config.NumberColumn("Target", format="₹%.2f"),
                        "SL": st.column_config.NumberColumn("SL", format="₹%.2f"),
                        "Support": st.column_config.NumberColumn("Support", format="₹%.2f"),
                        "Resist": st.column_config.NumberColumn("Resist", format="₹%.2f"),
                        "RSI": st.column_config.NumberColumn("RSI", format="%.2f"),
                        "Signal": st.column_config.TextColumn("Signal")
                    }
                )

# ═══════════════════════════════════════════════════════════════════════════════
# Utility Tabs
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    a = calc_analytics(df)
    if not a or a.get("closed_trades", 0) == 0:
        st.info("Close some trades to see analytics. Here's portfolio overview:")
        if not df.empty:
            st.dataframe(df[["stock", "quantity", "buy_at", "cmp", "profit", "profit_pct", "status"]],
                         use_container_width=True)
    else:
        st.markdown('<div class="sec">Performance Metrics</div>', unsafe_allow_html=True)
        st.markdown('<div class="cards">' + card("Win Rate", f'{a["win_rate"]}%', f'{a["wins"]}W / {a["losses"]}L', "green" if a["win_rate"] >= 50 else "red") + card("Profit Factor", str(a["profit_factor"]), "Gross P / Gross L") + card("Expectancy", f'₹{a["expectancy"]}') + card("Avg Win", f'₹{a["avg_win"]:,.0f}') + card("Avg Loss", f'₹{a["avg_loss"]:,.0f}', "", "red") + card("Max Drawdown", f'₹{a["max_drawdown"]:,.0f}', "", "red") + card("Avg Hold", f'{a["avg_hold_days"]}d') + card("Sharpe", str(a["sharpe"])) + '</div>', unsafe_allow_html=True)

with tab7:
    st.markdown('<div class="sec">👁 Watchlist</div>', unsafe_allow_html=True)
    wdf = get_watchlist()
    if not wdf.empty:
        st.dataframe(wdf, use_container_width=True, hide_index=True)
    else:
        st.caption("No stocks in watchlist yet.")

with tab8:
    if df.empty:
        st.info("No data to export.")
    else:
        st.markdown('<div class="sec">Export Controls</div>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)