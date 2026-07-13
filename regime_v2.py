"""
regime_v2.py — Regime intelligence layer (additive).

The base regime in signals.py reads ONLY the Nifty 50 EMA structure. But the
dashboard already FETCHES India VIX, Midcap, Smallcap and Bank Nifty every
cycle — and then throws them away. This module puts that data to work, and adds
the two things a swing trader actually needs beyond "bull or bear":

  1. VOLATILITY REGIME (India VIX)
     Direction alone is not enough. A "Bull" tape with VIX at 22 behaves nothing
     like a "Bull" tape with VIX at 11 — stops get hit, gaps widen, and breakout
     failure rates climb. VIX tells you HOW MUCH RISK to take, which the base
     regime never answered.

  2. TREND STRENGTH vs CHOP (ADX)
     This is the single biggest killer of swing systems. A "Bull" regime that is
     actually CHOPPING sideways produces breakout after breakout that fails.
     ADX separates a real trend from a directionless drift — the base regime
     cannot tell them apart, because EMAs stay stacked in a chop.

  3. INDEX DIVERGENCE (Nifty vs Midcap vs Smallcap)
     When the index is up but the broader market isn't following, the rally is
     narrow and fragile. This is classic late-stage behaviour.

  4. REGIME AGE + TRANSITION
     "Bull" on day 2 after a flip is a different trade from "Bull" on day 60.
     Fresh regimes are where the trend-following edge lives; old ones are where
     it dies.

Everything degrades gracefully: any missing index simply drops out of the
calculation rather than breaking the regime.
"""

import numpy as np
import pandas as pd


# ── 1. Volatility regime (India VIX) ─────────────────────────────────────────
# Indian VIX bands. Long-run median sits ~13-15; sustained >20 is genuine stress.
def volatility_regime(vix_df):
    """Returns dict with vix level, band, percentile and a risk instruction."""
    if vix_df is None or len(vix_df) < 20:
        return None
    try:
        close = vix_df["Close"].dropna()
        vix = float(close.iloc[-1])
        # where does today's VIX sit vs its own last year?
        lookback = close.tail(252)
        pct = float((lookback < vix).sum() / len(lookback) * 100)
        prev = float(close.iloc[-2]) if len(close) >= 2 else vix
        chg = (vix / prev - 1) * 100 if prev else 0.0
    except Exception:
        return None

    if vix < 12:
        band, risk = "Complacent", "Full size OK — but complacency precedes shocks"
        color = "#10b981"
    elif vix < 15:
        band, risk = "Calm", "Normal position size"
        color = "#10b981"
    elif vix < 20:
        band, risk = "Elevated", "Trim size ~25% · widen stops (noise is larger)"
        color = "#f59e0b"
    elif vix < 26:
        band, risk = "High", "Half size · breakouts fail more often here"
        color = "#ef4444"
    else:
        band, risk = "Panic", "Stand aside or minimal size · gaps can jump stops"
        color = "#dc2626"

    return {"vix": round(vix, 2), "band": band, "percentile": round(pct),
            "change_pct": round(chg, 1), "risk_note": risk, "color": color,
            "spiking": bool(chg > 10)}


# ── 2. Trend strength vs chop (ADX) ──────────────────────────────────────────
def _adx(high, low, close, period=14):
    """Wilder's ADX. Returns the latest value, or None."""
    try:
        h = pd.Series(high, dtype=float).reset_index(drop=True)
        l = pd.Series(low, dtype=float).reset_index(drop=True)
        c = pd.Series(close, dtype=float).reset_index(drop=True)
        if len(c) < period * 3:
            return None

        up = h.diff()
        dn = -l.diff()
        plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
        minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)

        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                       axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1 / period, adjust=False).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1 / period, adjust=False).mean() / atr)

        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        val = adx.iloc[-1]
        return float(val) if not pd.isna(val) else None
    except Exception:
        return None


def trend_strength(nifty_df):
    """Is the index actually TRENDING, or just drifting sideways?

    This matters more than direction for a breakout system: in a chop, EMAs stay
    stacked (so the regime still says "Bull") while every breakout fails.
    """
    if nifty_df is None or len(nifty_df) < 50:
        return None
    adx = _adx(nifty_df["High"], nifty_df["Low"], nifty_df["Close"])
    if adx is None:
        return None

    if adx >= 30:
        state, note = "Strong Trend", "Trend-following works · let winners run"
        color, tradeable = "#10b981", True
    elif adx >= 22:
        state, note = "Trending", "Breakouts have follow-through"
        color, tradeable = "#10b981", True
    elif adx >= 16:
        state, note = "Weak Trend", "Mixed — be selective, take profits earlier"
        color, tradeable = "#f59e0b", True
    else:
        state, note = "Choppy / Rangebound", \
            "⚠️ Breakouts FAIL here. Mean-reversion, not momentum. Reduce activity."
        color, tradeable = "#ef4444", False

    return {"adx": round(adx, 1), "state": state, "note": note,
            "color": color, "trend_tradeable": tradeable}


# ── 3. Index divergence (is the whole market participating?) ─────────────────
def _ret(df, bars):
    try:
        c = df["Close"].dropna()
        if len(c) <= bars:
            return None
        return float((c.iloc[-1] / c.iloc[-1 - bars] - 1) * 100)
    except Exception:
        return None


def breadth_divergence(bulk, lookback=21):
    """Nifty vs Midcap vs Smallcap over the last month.

    A rally the broader market refuses to join is a NARROW rally — historically
    the late stage of a move. This is the cheapest breadth read available, and
    the data is already being fetched.
    """
    nifty = bulk.get("^NSEI")
    mid = bulk.get("^NSEMDCP50")
    small = bulk.get("^CNXSC")
    if nifty is None:
        return None

    n_ret = _ret(nifty, lookback)
    m_ret = _ret(mid, lookback) if mid is not None else None
    s_ret = _ret(small, lookback) if small is not None else None
    if n_ret is None or (m_ret is None and s_ret is None):
        return None

    broad = [x for x in (m_ret, s_ret) if x is not None]
    broad_avg = sum(broad) / len(broad)
    gap = broad_avg - n_ret          # +ve = broad market LEADING (healthy)

    if n_ret > 0 and gap < -3:
        state = "Narrow Rally"
        note = ("Nifty is rising but mid/smallcaps are lagging badly — the "
                "rally is carried by a few heavyweights. Fragile.")
        color, healthy = "#ef4444", False
    elif n_ret > 0 and gap > 1:
        state = "Broad Participation"
        note = "Healthy — the whole market is joining the move."
        color, healthy = "#10b981", True
    elif n_ret < 0 and gap > 3:
        state = "Selective Weakness"
        note = "Index is falling but broader market holds up — often a bottoming tell."
        color, healthy = "#f59e0b", True
    elif n_ret < 0 and gap < -3:
        state = "Broad Selloff"
        note = "Everything is being sold — capitulation risk, but also where bottoms form."
        color, healthy = "#ef4444", False
    else:
        state = "In Line"
        note = "Broad market moving with the index."
        color, healthy = "#6b7280", True

    return {"nifty_ret": round(n_ret, 1),
            "midcap_ret": round(m_ret, 1) if m_ret is not None else None,
            "smallcap_ret": round(s_ret, 1) if s_ret is not None else None,
            "gap": round(gap, 1), "state": state, "note": note,
            "color": color, "healthy": healthy}


# ── 4. Regime age & transition ───────────────────────────────────────────────
def regime_age(nifty_df):
    """How many sessions has price held its current side of the 200 EMA?

    A regime 3 days old is where trend-following pays; the same regime 90 days
    old is crowded and prone to reversal. The base regime is blind to this.
    """
    if nifty_df is None or len(nifty_df) < 200:
        return None
    try:
        close = nifty_df["Close"].dropna()
        ema200 = close.ewm(span=200, adjust=False).mean()
        above = (close > ema200).values
        if len(above) < 2:
            return None
        current = bool(above[-1])
        days = 1
        for i in range(len(above) - 2, -1, -1):
            if bool(above[i]) == current:
                days += 1
            else:
                break
    except Exception:
        return None

    if days <= 5:
        phase = "Fresh flip"
        note = "Regime just changed — highest trend-following edge, but confirm."
    elif days <= 25:
        phase = "Establishing"
        note = "Trend is young — the sweet spot for swing entries."
    elif days <= 90:
        phase = "Mature"
        note = "Trend is well established — still tradeable, tighten trailing stops."
    else:
        phase = "Extended"
        note = "Regime is old and crowded — reversals get sharper from here."

    return {"days": days, "side": "above 200EMA" if current else "below 200EMA",
            "phase": phase, "note": note, "just_flipped": days <= 5}


# ── 5. Composite: what should I actually DO? ─────────────────────────────────
def playbook(regime, vol, trend, div):
    """Turn the readings into concrete instructions. This is the part the base
    regime never gave: it told you the weather, not what to wear."""
    actions, warnings = [], []

    bullish = regime in ("Strong Bull", "Bull")
    bearish = regime in ("Strong Bear", "Bear")

    # Direction
    if bullish:
        actions.append("Long setups favoured — breakouts, VCP, pocket pivots")
    elif bearish:
        actions.append("Capital preservation — avoid new longs; cash is a position")
    elif regime == "Bull Pullback":
        actions.append("Uptrend intact but pulling back — buy strength off support, don't chase")
    elif regime == "Bear Rally":
        actions.append("Counter-trend bounce in a downtrend — this is a SELLING rally, not a bottom")
        warnings.append("Bear rallies are violent and fail — do not mistake this for a new bull")
    else:
        actions.append("No clear regime — reduce activity until direction resolves")

    # Volatility overrides direction
    if vol:
        if vol["band"] in ("High", "Panic"):
            warnings.append(f"VIX {vol['vix']} ({vol['band']}) — {vol['risk_note']}")
        elif vol["spiking"]:
            warnings.append(f"VIX spiked {vol['change_pct']}% today — something is breaking")
        else:
            actions.append(f"VIX {vol['vix']} ({vol['band']}) — {vol['risk_note']}")

    # Chop overrides everything for a breakout system
    if trend and not trend["trend_tradeable"]:
        warnings.append(
            f"ADX {trend['adx']} — market is CHOPPING. Breakout systems bleed here. "
            f"This is the #1 reason a good scanner still loses money.")

    # Participation
    if div and not div["healthy"]:
        warnings.append(div["note"])

    return {"actions": actions, "warnings": warnings}


def enrich_regime(base_regime_dict, bulk_data):
    """Take signals.get_market_regime()'s output + the fetched index frames, and
    return it enriched with volatility / trend-strength / divergence / age /
    playbook. Never raises — a failed sub-read just comes back as None."""
    out = dict(base_regime_dict or {})
    nifty = (bulk_data or {}).get("^NSEI")
    vix_df = (bulk_data or {}).get("^INDIAVIX")

    try:
        out["volatility"] = volatility_regime(vix_df)
    except Exception:
        out["volatility"] = None
    try:
        out["trend_strength"] = trend_strength(nifty)
    except Exception:
        out["trend_strength"] = None
    try:
        out["divergence"] = breadth_divergence(bulk_data or {})
    except Exception:
        out["divergence"] = None
    try:
        out["age"] = regime_age(nifty)
    except Exception:
        out["age"] = None
    try:
        out["playbook"] = playbook(out.get("regime", "Unknown"),
                                   out.get("volatility"),
                                   out.get("trend_strength"),
                                   out.get("divergence"))
    except Exception:
        out["playbook"] = {"actions": [], "warnings": []}
    return out
