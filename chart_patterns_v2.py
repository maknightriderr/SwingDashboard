"""
chart_patterns_v2.py — Strong daily-timeframe patterns (additive module).

Adds four high-value patterns the base engine was missing, each chosen for
swing/momentum trading on NSE daily bars:

  1. Inside Bar        — today's range fully inside yesterday's ("mother bar").
                         A coil; breakout above the mother-bar high is the signal.
  2. NR7 (Narrow Range)— today's range is the narrowest of the last 7 bars.
                         Volatility contraction that precedes expansion. An
                         "NR7 inside bar" (both at once) is a classic pre-breakout.
  3. Ascending Triangle— flat resistance (equal highs) + rising support (higher
                         lows). Bullish continuation; break above resistance.
  4. Descending Triangle—flat support (equal lows) + falling resistance (lower
                         highs). Bearish continuation; break below support.
  5. Pocket Pivot      — (O'Neil/Morales) an up-day inside a base whose volume
                         exceeds the highest DOWN-day volume of the prior 10
                         days, with price holding its key EMAs. Early
                         institutional-accumulation footprint before a breakout.

Everything is pure-function and degrades gracefully — any bad input just
returns "not detected" rather than raising. Designed to be called from
signals.py's compute_indicators with the same OHLCV series it already has.
"""

import numpy as np


def detect_inside_bar(high, low):
    """Today's bar fully inside yesterday's range. Returns dict with the
    mother-bar high/low (breakout / breakdown triggers)."""
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    if len(h) < 2:
        return {"inside_bar": False, "mother_high": None, "mother_low": None}
    is_inside = bool(h[-1] < h[-2] and l[-1] > l[-2])
    return {
        "inside_bar": is_inside,
        "mother_high": round(float(h[-2]), 2) if is_inside else None,
        "mother_low": round(float(l[-2]), 2) if is_inside else None,
    }


def detect_nr7(high, low):
    """Today's range is the narrowest of the last 7 bars (NR7)."""
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    if len(h) < 7:
        return {"nr7": False}
    ranges = (h[-7:] - l[-7:])
    is_nr7 = bool(int(np.argmin(ranges)) == 6)   # position 6 = today = narrowest
    return {"nr7": is_nr7}


def detect_triangles(high, low, close, atr, lookback=30, tol_atr=0.6):
    """Ascending / Descending triangle over the lookback window.

    Ascending: highs roughly EQUAL (flat resistance) while lows RISE.
    Descending: lows roughly EQUAL (flat support) while highs FALL.

    Uses simple linear fits on the recent swing highs & lows; 'flat' means the
    slope is within tol_atr*ATR across the window, 'rising'/'falling' means it
    clears that band. Returns flags + the horizontal level (breakout trigger).
    """
    out = {"ascending_triangle": False, "descending_triangle": False,
           "triangle_level": None}
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    n = len(c)
    if n < lookback or atr <= 0:
        return out

    hh = h[-lookback:]
    ll = l[-lookback:]
    x = np.arange(lookback)

    # Linear slope of highs and of lows across the window (price units per bar)
    hi_slope = np.polyfit(x, hh, 1)[0]
    lo_slope = np.polyfit(x, ll, 1)[0]

    # "flat" band: total drift across the window under tol_atr * ATR
    flat_band = (tol_atr * atr) / lookback

    hi_flat = abs(hi_slope) <= flat_band
    lo_flat = abs(lo_slope) <= flat_band
    hi_falling = hi_slope < -flat_band
    lo_rising = lo_slope > flat_band

    res_level = float(np.max(hh))
    sup_level = float(np.min(ll))

    # Ascending: flat resistance + rising support
    if hi_flat and lo_rising:
        out["ascending_triangle"] = True
        out["triangle_level"] = round(res_level, 2)   # break ABOVE this = trigger
    # Descending: flat support + falling resistance
    elif lo_flat and hi_falling:
        out["descending_triangle"] = True
        out["triangle_level"] = round(sup_level, 2)   # break BELOW this = trigger

    return out


def detect_pocket_pivot(close, vol, ema10=None, ema50=None, lookback=10):
    """O'Neil/Morales Pocket Pivot: an up-day whose volume is greater than the
    largest DOWN-day volume of the prior `lookback` days, ideally with price
    holding above/near its key moving averages (accumulation inside a base).

    Returns {pocket_pivot: bool, detail: str}.
    """
    c = np.asarray(close, dtype=float)
    v = np.asarray(vol, dtype=float)
    if len(c) < lookback + 2:
        return {"pocket_pivot": False, "detail": ""}

    up_day = c[-1] > c[-2]
    if not up_day:
        return {"pocket_pivot": False, "detail": ""}

    # Largest down-day volume over the prior `lookback` days (excluding today)
    prior_c = c[-(lookback + 1):-1]
    prior_v = v[-(lookback + 1):-1]
    down_vols = [prior_v[i] for i in range(1, len(prior_c))
                 if prior_c[i] < prior_c[i - 1]]
    if not down_vols:
        return {"pocket_pivot": False, "detail": ""}

    max_down_vol = max(down_vols)
    today_vol = v[-1]
    vol_ok = today_vol > max_down_vol

    # Price should be holding its base — near/above the 10 & 50 EMAs if provided
    cmp = float(c[-1])
    ma_ok = True
    if ema10 is not None and ema50 is not None:
        # within ~2% below the 10EMA (not extended far under) and above 50EMA
        ma_ok = cmp >= ema10 * 0.98 and cmp >= ema50 * 0.98

    if vol_ok and ma_ok:
        return {"pocket_pivot": True,
                "detail": f"up-day vol {today_vol/max_down_vol:.1f}x largest "
                          f"prior down-day"}
    return {"pocket_pivot": False, "detail": ""}


def detect_coiling(high, low, close, recent_n=5, base_n=25):
    """Volatility compression over recent bars vs the base's typical range.
    A VCP-style coiled base is tight over WEEKS but rarely has a single bar that
    meets the strict Inside Bar / NR7 definition, so those miss it. This fires
    when the average bar range of the last `recent_n` bars has contracted to
    <=60% of the `base_n`-bar average, while price is still holding (not breaking
    down). Gives coiled bases a meaningful compression tag."""
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    if len(h) < base_n + 1:
        return {"coiling": False, "coil_pct": None}
    recent = (h[-recent_n:] - l[-recent_n:]).mean()
    base = (h[-base_n:] - l[-base_n:]).mean()
    if base <= 0:
        return {"coiling": False, "coil_pct": None}
    ratio = recent / base
    holding = c[-1] >= c[-recent_n:].min() * 0.98
    is_coiling = bool(ratio <= 0.6 and holding)
    return {"coiling": is_coiling,
            "coil_pct": round((1 - ratio) * 100, 1) if is_coiling else None}


def detect_all(open_p, high, low, close, vol, atr, ema10=None, ema50=None):
    """Convenience aggregator — runs every v2 pattern and returns a flat dict
    plus a list of human-readable pattern tags for display."""
    res = {}
    res.update(detect_inside_bar(high, low))
    res.update(detect_nr7(high, low))
    res.update(detect_triangles(high, low, close, atr))
    res.update(detect_pocket_pivot(close, vol, ema10, ema50))
    res.update(detect_coiling(high, low, close))

    tags = []
    # NR7 + inside bar together is the strongest compression signal
    if res.get("inside_bar") and res.get("nr7"):
        tags.append("🎯 NR7 Inside Bar (coiled)")
    elif res.get("inside_bar"):
        tags.append("📦 Inside Bar")
    elif res.get("nr7"):
        tags.append("🤏 NR7 (narrow range)")
    if res.get("ascending_triangle"):
        tags.append("📐 Ascending Triangle")
    if res.get("descending_triangle"):
        tags.append("📐 Descending Triangle")
    if res.get("pocket_pivot"):
        tags.append("💰 Pocket Pivot")
    # Coiling — only add if no stronger single-bar compression tag already fired
    if res.get("coiling") and not (res.get("inside_bar") or res.get("nr7")):
        _cp = res.get("coil_pct")
        tags.append(f"🎯 Coiling ({_cp:.0f}% tighter)" if _cp else "🎯 Coiling")
    res["v2_tags"] = tags
    return res


if __name__ == "__main__":
    import pandas as pd
    # quick smoke test
    n = 40
    close = np.linspace(100, 110, n)
    high = close + 1
    low = close - 1
    vol = np.full(n, 1e6)
    r = detect_all(pd.Series(close - 0.2), pd.Series(high), pd.Series(low),
                   pd.Series(close), pd.Series(vol), atr=1.5,
                   ema10=109, ema50=105)
    print(r)
