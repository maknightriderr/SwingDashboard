"""
theme_scanner.py — NSE Market Theme Scanner
============================================
Groups NSE stocks into curated MARKET THEMES (Defence, Railways, EV, PSU Banks,
Green/Renewable Energy, Capex/Infra, etc.) — narratives that cut ACROSS the
official sectors in your universe CSVs — then scores and ranks each theme by
aggregate strength so you can see where capital is rotating.

Why curated baskets?
--------------------
Your universe CSVs carry official *sectors* (e.g. "Capital Goods", "Power")
but not market *themes* like "Defence" or "Railways" — those are cross-sector
narratives the market trades. So the theme→stock mapping below is hand-curated
from well-known NSE theme constituents. At load time we KEEP ONLY the symbols
that actually exist in your universe (signals.SECTOR_MAP), so nothing references
a stock you don't track. You can freely edit THEME_BASKETS to taste.

Public API
----------
get_available_themes()              -> list[str]
scan_themes(min_constituents=3)     -> dict  (ranked themes + per-theme detail)

scan_themes returns:
{
  "themes": [ {theme, score, rank, n, n_scored, avg_ret_1m, avg_ret_3m,
               pct_above_50ema, pct_above_200ema, avg_rs, breadth_up,
               leaders:[{stock, ret_1m, ret_3m, rs_ratio, trend, above_50ema,
                         above_200ema, cmp}] }, ... ],   # sorted hottest->coldest
  "scanned": int, "themes_count": int, "timestamp": str,
}

All network work reuses signals._bulk_fetch_history + signals.compute_indicators,
so it shares the same caches and degrades gracefully (never raises).
"""

from datetime import datetime

import numpy as np

import signals as _sg


# ==============================================================================
# CURATED NSE THEME BASKETS
# ------------------------------------------------------------------------------
# Seeded from well-known NSE theme constituents. At import we filter each list
# down to only the symbols present in your universe (signals.SECTOR_MAP), so the
# scanner never references a stock you don't actually track. Edit freely.
# ==============================================================================
THEME_BASKETS = {
    "Defence": [
        "HAL", "BEL", "BDL", "MAZDOCK", "COCHINSHIP", "GRSE", "MIDHANI",
        "DATAPATTNS", "PARAS", "ASTRAMICRO", "ZENTEC", "DCXINDIA", "IDEAFORGE",
        "BEML", "SOLARINDS", "MTARTECH",
    ],
    "Railways": [
        "IRCTC", "IRFC", "RVNL", "IRCON", "RAILTEL", "TITAGARH", "JWL",
        "TEXRAIL", "RITES", "CONCOR", "BEML", "JUPITER", "HBLENGINE",
    ],
    "PSU Banks": [
        "SBIN", "BANKBARODA", "PNB", "CANBK", "UNIONBANK", "INDIANB",
        "BANKINDIA", "CENTRALBK", "IOB", "UCOBANK", "MAHABANK", "PSB",
        "J&KBANK",
    ],
    "Green / Renewable Energy": [
        "ADANIGREEN", "TATAPOWER", "JSWENERGY", "NTPC", "POWERGRID",
        "SUZLON", "INOXWIND", "WAAREEENER", "PREMIERENE", "NHPC", "SJVN",
        "IREDA", "ADANIENSOL", "BHEL", "KPIGREEN", "ORIENTGREEN",
    ],
    "EV / Auto Electrification": [
        "TATAMOTORS", "M&M", "OLECTRA", "TATAPOWER", "EXIDEIND", "AMARAJABAT",
        "AMARARAJA", "SONACOMS", "UNOMINDA", "BOSCHLTD", "BHARATFORG",
        "MOTHERSON", "GREAVESCOT", "JBMA", "HBLENGINE",
    ],
    "Capex / Infrastructure": [
        "LT", "GMRINFRA", "GMRAIRPORT", "IRB", "KEC", "KALPATPOWR",
        "KPIL", "NCC", "NBCC", "PNCINFRA", "HGINFRA", "ASHOKA", "GPIL",
        "RVNL", "IRCON", "AFCONS", "CONCOR",
    ],
    "Power / Utilities": [
        "NTPC", "POWERGRID", "TATAPOWER", "JSWENERGY", "ADANIPOWER",
        "ADANIENSOL", "NHPC", "SJVN", "TORNTPOWER", "CESC", "NLCINDIA",
        "BHEL", "THERMAX", "POWERINDIA", "GVT&D",
    ],
    "PSU (ex-Banks)": [
        "COALINDIA", "ONGC", "IOC", "BPCL", "GAIL", "POWERGRID", "NTPC",
        "HAL", "BEL", "NMDC", "NLCINDIA", "SAIL", "RVNL", "IRCON",
        "CONCOR", "BHEL", "OIL", "HUDCO", "RECLTD", "PFC", "IRFC",
    ],
    "IT / Technology": [
        "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM", "PERSISTENT",
        "COFORGE", "MPHASIS", "LTTS", "KPITTECH", "TATAELXSI", "OFSS",
        "ZENSARTECH", "BSOFT", "CYIENT",
    ],
    "Pharma / Healthcare": [
        "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "LUPIN", "AUROPHARMA",
        "ZYDUSLIFE", "ALKEM", "TORNTPHARM", "BIOCON", "GLENMARK",
        "LAURUSLABS", "MANKIND", "ABBOTINDIA", "APOLLOHOSP", "FORTIS",
        "MAXHEALTH",
    ],
    "Banking / Financials (Private)": [
        "HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK", "INDUSINDBK",
        "BAJFINANCE", "BAJAJFINSV", "SBICARD", "CHOLAFIN", "SHRIRAMFIN",
        "IDFCFIRSTB", "FEDERALBNK", "BANDHANBNK", "AUBANK", "PNBHOUSING",
    ],
    "FMCG / Consumption": [
        "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "MARICO",
        "GODREJCP", "COLPAL", "TATACONSUM", "VBL", "UNITDSPR", "RADICO",
        "PGHH", "EMAMILTD", "JYOTHYLAB",
    ],
    "Metals / Mining": [
        "TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "JINDALSTEL", "SAIL",
        "NMDC", "NATIONALUM", "HINDZINC", "HINDCOPPER", "APLAPOLLO",
        "JSL", "RATNAMANI", "WELCORP", "COALINDIA",
    ],
    "Real Estate": [
        "DLF", "GODREJPROP", "LODHA", "OBEROIRLTY", "PRESTIGE", "PHOENIXLTD",
        "BRIGADE", "SOBHA", "MAHLIFE", "SUNTECK", "ANANTRAJ", "RAYMOND",
        "NBCC", "HUDCO",
    ],
    "New-Age / Platform": [
        "ZOMATO", "ETERNAL", "NYKAA", "PAYTM", "POLICYBZR", "DELHIVERY",
        "PBFINTECH", "MAPMYINDIA", "INDIAMART", "JUSTDIAL", "IRCTC",
        "CARTRADE", "EASEMYTRIP", "NAUKRI",
    ],
}


def _norm(sym: str) -> str:
    """Normalise a symbol the same way signals.py does (strip suffixes/upper)."""
    s = str(sym).upper().strip()
    for sfx in (".NS", ".BO", ".NSE", ".BSE"):
        if s.endswith(sfx):
            s = s[: -len(sfx)]
    return s


def _build_filtered_baskets():
    """Keep only symbols that exist in the loaded universe (SECTOR_MAP).
    Returns (filtered_baskets, all_symbols_set)."""
    universe = set(_sg.SECTOR_MAP.keys())
    filtered = {}
    all_syms = set()
    for theme, syms in THEME_BASKETS.items():
        present = []
        seen = set()
        for s in syms:
            n = _norm(s)
            if n in universe and n not in seen:
                present.append(n)
                seen.add(n)
                all_syms.add(n)
        if present:
            filtered[theme] = present
    return filtered, all_syms


# Build once at import. If the universe isn't loaded yet for some reason, this
# simply yields empty baskets and scan_themes will report nothing (no crash).
THEME_CONSTITUENTS, _ALL_THEME_SYMBOLS = _build_filtered_baskets()


def get_available_themes():
    """Return the list of themes that have at least one universe constituent."""
    return sorted(THEME_CONSTITUENTS.keys())


def theme_coverage():
    """Diagnostic: how many curated names survived the universe filter per theme."""
    lines = ["🎯 Theme basket coverage (constituents present in your universe):"]
    for theme in sorted(THEME_BASKETS.keys()):
        curated = len(THEME_BASKETS[theme])
        present = len(THEME_CONSTITUENTS.get(theme, []))
        lines.append(f"  {theme:32s} {present:>3}/{curated:<3} present")
    lines.append(f"\nTotal distinct theme symbols in universe: {len(_ALL_THEME_SYMBOLS)}")
    return "\n".join(lines)


def _safe_pct(numer, denom):
    return round(numer / denom * 100, 1) if denom else 0.0


def scan_themes(min_constituents=3, period="6mo"):
    """
    Score every theme by aggregate strength and rank hottest -> coldest.

    For each constituent we reuse signals.compute_indicators (shared cache), then
    aggregate per theme:
      • avg_ret_1m / avg_ret_3m  — average constituent momentum
      • pct_above_50ema / pct_above_200ema — breadth above key EMAs
      • avg_rs   — average Relative-Strength ratio vs Nifty (1.0 = in line)
      • breadth_up — % of constituents in an uptrend
    Composite score blends momentum, breadth, RS and trend into one 0-100-ish
    figure used for ranking.

    Args:
        min_constituents : skip themes with fewer than this many scored stocks
        period           : history window for indicator computation

    Returns dict (see module docstring). Never raises.
    """
    baskets = THEME_CONSTITUENTS
    if not baskets:
        return {"themes": [], "scanned": 0, "themes_count": 0,
                "timestamp": datetime.now().strftime("%d %b %Y %H:%M"),
                "note": "No theme constituents found in the loaded universe."}

    # One bulk network pass for every distinct theme symbol (shared cache).
    all_syms = sorted(_ALL_THEME_SYMBOLS)
    try:
        bulk = _sg._bulk_fetch_history(all_syms, period=period)
    except Exception:
        bulk = {}

    # Compute indicators once per symbol, memoised in this call.
    ind_cache = {}

    def _ind(sym):
        if sym in ind_cache:
            return ind_cache[sym]
        try:
            res = _sg.compute_indicators(sym, period=period,
                                         prefetched_df=bulk.get(sym))
        except Exception:
            res = None
        ind_cache[sym] = res
        return res

    theme_rows = []
    scanned = 0

    for theme, syms in baskets.items():
        rets_1m, rets_3m, rs_vals = [], [], []
        above50 = above200 = up_trend = 0
        n_scored = 0
        leaders = []

        for sym in syms:
            ind = _ind(sym)
            if not ind:
                continue
            scanned += 1
            n_scored += 1
            cmp = ind.get("cmp")

            # 1M / 3M returns from RS periods if available, else from EMAs proxy
            r1 = r3 = None
            periods = ind.get("rs_periods") or {}
            if periods:
                r1 = periods.get("21", {}).get("stock")
                r3 = periods.get("63", {}).get("stock")
            if r1 is not None:
                rets_1m.append(r1)
            if r3 is not None:
                rets_3m.append(r3)

            # Breadth vs EMAs
            ema50 = ind.get("ema50")
            ema200 = ind.get("ema200")
            a50 = bool(cmp and ema50 and cmp >= ema50)
            a200 = bool(cmp and ema200 and cmp >= ema200)
            if a50:
                above50 += 1
            if a200:
                above200 += 1

            # Trend breadth
            trend = ind.get("trend", "")
            if "Uptrend" in str(trend):
                up_trend += 1

            # Relative strength
            rs = ind.get("rs_ratio")
            if rs is not None:
                rs_vals.append(rs)

            leaders.append({
                "stock": sym,
                "cmp": cmp,
                "ret_1m": round(r1, 1) if r1 is not None else None,
                "ret_3m": round(r3, 1) if r3 is not None else None,
                "rs_ratio": round(rs, 2) if rs is not None else None,
                "trend": trend,
                "above_50ema": a50,
                "above_200ema": a200,
            })

        if n_scored < min_constituents:
            continue

        avg_ret_1m = round(float(np.mean(rets_1m)), 1) if rets_1m else 0.0
        avg_ret_3m = round(float(np.mean(rets_3m)), 1) if rets_3m else 0.0
        avg_rs = round(float(np.mean(rs_vals)), 3) if rs_vals else 1.0
        pct_above_50 = _safe_pct(above50, n_scored)
        pct_above_200 = _safe_pct(above200, n_scored)
        breadth_up = _safe_pct(up_trend, n_scored)

        # ── Composite theme strength score ────────────────────────────────────
        # Blend (weights chosen so a hot theme lands ~60-90):
        #   momentum  : 1M (x1.2) + 3M (x0.8), capped
        #   breadth   : % above 50EMA (x0.35) + % above 200EMA (x0.15)
        #   trend     : % in uptrend (x0.20)
        #   RS        : (avg_rs - 1) scaled — leaders above Nifty add, laggards subtract
        mom = avg_ret_1m * 1.2 + avg_ret_3m * 0.8
        mom = max(-40.0, min(60.0, mom))                 # clamp outliers
        breadth = pct_above_50 * 0.35 + pct_above_200 * 0.15
        trend_component = breadth_up * 0.20
        rs_component = (avg_rs - 1.0) * 60.0              # +0.1 RS ≈ +6 pts
        rs_component = max(-25.0, min(25.0, rs_component))

        score = mom + breadth + trend_component + rs_component
        # Clamp to a clean 0-100 range so the UI strength bar renders sensibly.
        score = round(max(0.0, min(100.0, score)), 1)

        # Rank constituents within theme: RS first, then 3M, then 1M
        leaders.sort(
            key=lambda x: (
                x["rs_ratio"] if x["rs_ratio"] is not None else -9,
                x["ret_3m"] if x["ret_3m"] is not None else -999,
                x["ret_1m"] if x["ret_1m"] is not None else -999,
            ),
            reverse=True,
        )

        theme_rows.append({
            "theme": theme,
            "score": score,
            "n": len(syms),
            "n_scored": n_scored,
            "avg_ret_1m": avg_ret_1m,
            "avg_ret_3m": avg_ret_3m,
            "pct_above_50ema": pct_above_50,
            "pct_above_200ema": pct_above_200,
            "avg_rs": avg_rs,
            "breadth_up": breadth_up,
            "leaders": leaders,
        })

    # Rank themes hottest -> coldest
    theme_rows.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(theme_rows, 1):
        r["rank"] = i

    return {
        "themes": theme_rows,
        "scanned": scanned,
        "themes_count": len(theme_rows),
        "timestamp": datetime.now().strftime("%d %b %Y %H:%M"),
    }
