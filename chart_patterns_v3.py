"""
chart_patterns_v3.py — the remaining high-value swing patterns (additive).

Fills the gaps left by detect_price_patterns() and chart_patterns_v2:

  BULLISH
    Falling Wedge breakout   — converging DOWN-sloping highs & lows, then a
                               break upward. A classic Tier-2 reversal.
    Bullish Pennant          — sharp up-pole, then a small SYMMETRICAL
                               consolidation (unlike a flag, which slopes down).
    Rectangle Breakout       — horizontal range (flat top AND flat bottom)
                               resolved upward on expanding volume.
    Rounding Bottom          — slow saucer base: falling first half, rising
                               second half, shallow curve.

  BEARISH
    Rising Wedge breakdown   — converging UP-sloping highs & lows, then a break
                               downward. Deceptive: looks bullish while forming.
    Bear Flag                — sharp down-pole, then a tight UPWARD drift, then
                               continuation down.
    Bearish Pennant          — down-pole then symmetrical squeeze, resolving down.

  NEUTRAL-UNTIL-RESOLVED
    Symmetrical Triangle     — converging highs and lows with no directional
                               bias until price actually breaks one side.

DESIGN
------
Every detector is pure-function, tolerant of bad input (returns "not detected"
instead of raising), and requires an actual BREAK where the pattern implies one
— a shape alone is not a signal. Volume expansion is required for the breakout
patterns, because an unconfirmed break is the single most common false positive.
"""

import numpy as np


def _slope(y):
    """Least-squares slope of a series (price units per bar)."""
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return 0.0
    x = np.arange(len(y))
    return float(np.polyfit(x, y, 1)[0])


def _swing_highs(h, l, atr, min_gap=3):
    """Indices of local swing highs (fractals with 2 bars either side)."""
    out = []
    for i in range(2, len(h) - 2):
        if (h[i] >= h[i-1] and h[i] >= h[i-2]
                and h[i] >= h[i+1] and h[i] >= h[i+2]):
            if not out or i - out[-1] >= min_gap:
                out.append(i)
    return out


def _swing_lows(h, l, atr, min_gap=3):
    out = []
    for i in range(2, len(l) - 2):
        if (l[i] <= l[i-1] and l[i] <= l[i-2]
                and l[i] <= l[i+1] and l[i] <= l[i+2]):
            if not out or i - out[-1] >= min_gap:
                out.append(i)
    return out


def _fit_line(y):
    """Least-squares fit; returns (slope, projected value at the last index)."""
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return 0.0, float(y[-1]) if len(y) else 0.0
    x = np.arange(len(y))
    m, b = np.polyfit(x, y, 1)
    return float(m), float(m * (len(y) - 1) + b)


def detect_wedges(high, low, close, vol, vol_avg, lookback=35):
    """Falling Wedge (bullish) and Rising Wedge (bearish).

    A wedge needs BOTH boundaries sloping the SAME way while CONVERGING:
      falling wedge : highs down, lows down, gap narrowing -> break UP
      rising wedge  : highs up,   lows up,   gap narrowing -> break DOWN

    The breakout is measured against the PROJECTED TRENDLINE, not against the
    recent highest high. That distinction matters: a falling wedge's upper
    boundary is sloping DOWN, so price can break the trendline while still
    sitting below older highs — testing against the raw max would miss almost
    every real wedge breakout.
    """
    out = {"falling_wedge": False, "rising_wedge": False, "wedge_level": None}
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    if len(c) < lookback + 3:
        return out

    hh, ll = h[-lookback:], l[-lookback:]
    hs, upper_at_end = _fit_line(hh)
    ls, lower_at_end = _fit_line(ll)

    n3 = max(4, lookback // 3)
    early = float(np.mean(hh[:n3] - ll[:n3]))
    late = float(np.mean(hh[-n3:] - ll[-n3:]))
    if early <= 0:
        return out
    converging = late < early * 0.72

    cmp_ = float(c[-1])
    vr = (float(vol[-1]) / vol_avg) if vol_avg else 1.0

    if converging and hs < 0 and ls < 0:
        # Bullish only once price closes above the descending upper trendline
        if cmp_ > upper_at_end and vr >= 1.2:
            out["falling_wedge"] = True
            out["wedge_level"] = round(upper_at_end, 2)
    elif converging and hs > 0 and ls > 0:
        # Bearish once price closes below the rising lower trendline
        if cmp_ < lower_at_end:
            out["rising_wedge"] = True
            out["wedge_level"] = round(lower_at_end, 2)
    return out


def detect_pennants_and_bear_flag(high, low, close, vol, vol_avg,
                                  pole_bars=10, cons_bars=8):
    """Bullish/Bearish Pennant and Bear Flag.

    Pennant  = sharp pole + SYMMETRICAL squeeze (highs down, lows up).
    Bear flag= sharp DOWN pole + tight UPWARD drift, then continuation down.
    All require the pole to be a genuine impulse (>=8%) and the consolidation to
    be tight relative to it, otherwise any drift qualifies.
    """
    out = {"bullish_pennant": False, "bearish_pennant": False,
           "bear_flag": False, "pole_pct": None}
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    need = pole_bars + cons_bars + 2
    if len(c) < need:
        return out

    pole = c[-(pole_bars + cons_bars):-cons_bars]
    cons_h = h[-cons_bars:]
    cons_l = l[-cons_bars:]
    cons_c = c[-cons_bars:]
    if len(pole) < 4:
        return out

    pole_gain = (pole[-1] - pole[0]) / max(abs(pole[0]), 1e-9) * 100
    cmp_ = float(c[-1])
    vr = (float(vol[-1]) / vol_avg) if vol_avg else 1.0

    # Measure the consolidation EXCLUDING the breakout bar. Including it drags
    # the drift slope the wrong way — a bear flag's upward drift reads as
    # "falling" once the breakdown bar is averaged in, so the pattern never
    # fires on the very bar it completes.
    _ch, _cl, _cc = cons_h[:-1], cons_l[:-1], cons_c[:-1]
    if len(_cc) < 3:
        return out
    hs, ls = _slope(_ch), _slope(_cl)
    cons_range = float(np.max(_ch) - np.min(_cl))
    pole_range = float(abs(pole[-1] - pole[0]))
    tight = pole_range > 0 and cons_range < pole_range * 0.55
    symmetrical = hs < 0 and ls > 0          # both boundaries converging

    out["pole_pct"] = round(float(pole_gain), 2)

    # Bullish pennant: strong up-pole, symmetrical squeeze, breaks up on volume
    if pole_gain >= 8 and tight and symmetrical:
        if cmp_ > float(np.max(cons_h[:-1])) and vr >= 1.2:
            out["bullish_pennant"] = True

    # Bearish pennant: strong down-pole, symmetrical squeeze, breaks down
    if pole_gain <= -8 and tight and symmetrical:
        if cmp_ < float(np.min(cons_l[:-1])):
            out["bearish_pennant"] = True

    # Bear flag: down-pole then a tight UPWARD drift, then breaking down again
    if pole_gain <= -8 and tight and _slope(_cc) > 0:
        if cmp_ < float(np.min(cons_l[:-1])):
            out["bear_flag"] = True
    return out


def detect_rectangle(high, low, close, vol, vol_avg, lookback=30, tol=0.03):
    """Rectangle / range breakout: a horizontal band (flat resistance AND flat
    support) that price finally closes above on expanding volume."""
    out = {"rectangle_breakout": False, "rect_high": None, "rect_low": None}
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    if len(c) < lookback + 2:
        return out

    band_h = h[-(lookback + 1):-1]
    band_l = l[-(lookback + 1):-1]
    top, bot = float(np.max(band_h)), float(np.min(band_l))
    if top <= bot:
        return out

    # "Flat" = the touches cluster near the extremes rather than trending
    near_top = band_h[band_h >= top * (1 - tol)]
    near_bot = band_l[band_l <= bot * (1 + tol)]
    flat_enough = len(near_top) >= 3 and len(near_bot) >= 3
    height_pct = (top - bot) / bot * 100
    sane_height = 3 <= height_pct <= 25      # not a hair-thin or huge "range"

    cmp_ = float(c[-1])
    vr = (float(vol[-1]) / vol_avg) if vol_avg else 1.0
    if flat_enough and sane_height and cmp_ > top and vr >= 1.3:
        out["rectangle_breakout"] = True
        out["rect_high"] = round(top, 2)
        out["rect_low"] = round(bot, 2)
    return out


def detect_symmetrical_triangle(high, low, close, lookback=30):
    """Symmetrical triangle: highs falling AND lows rising (converging), with
    no directional bias until it breaks. Reported as a WATCH, not a signal."""
    out = {"symmetrical_triangle": False, "sym_high": None, "sym_low": None}
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    if len(h) < lookback + 2:
        return out
    hh, ll = h[-lookback:], l[-lookback:]
    hs, ls = _slope(hh), _slope(ll)
    n3 = max(4, lookback // 3)
    early = float(np.mean(hh[:n3] - ll[:n3]))
    late = float(np.mean(hh[-n3:] - ll[-n3:]))
    if early > 0 and late < early * 0.7 and hs < 0 and ls > 0:
        out["symmetrical_triangle"] = True
        out["sym_high"] = round(float(np.max(hh[-n3:])), 2)
        out["sym_low"] = round(float(np.min(ll[-n3:])), 2)
    return out


def detect_rounding_bottom(close, lookback=60):
    """Rounding (saucer) bottom: price falls through the first half, bottoms in
    the middle, and rises through the second half — a slow, shallow curve. The
    depth is capped so a V-crash-and-rebound doesn't qualify."""
    out = {"rounding_bottom": False, "saucer_low": None}
    c = np.asarray(close, dtype=float)
    if len(c) < lookback:
        return out
    seg = c[-lookback:]
    third = lookback // 3
    first, last = seg[:third], seg[2*third:]
    low_i = int(np.argmin(seg))
    bottom_in_middle = third * 0.6 <= low_i <= lookback - third * 0.6
    falling_then_rising = _slope(first) < 0 and _slope(last) > 0
    depth = (float(np.max(seg)) - float(np.min(seg))) / max(float(np.max(seg)), 1e-9) * 100
    shallow_curve = 8 <= depth <= 45
    near_highs = seg[-1] >= float(np.max(seg)) * 0.93
    if bottom_in_middle and falling_then_rising and shallow_curve and near_highs:
        out["rounding_bottom"] = True
        out["saucer_low"] = round(float(np.min(seg)), 2)
    return out


def detect_all_v3(high, low, close, vol, vol_avg):
    """Run every v3 detector; return flags plus display tags."""
    res = {}
    try:
        res.update(detect_wedges(high, low, close, vol, vol_avg))
        res.update(detect_pennants_and_bear_flag(high, low, close, vol, vol_avg))
        res.update(detect_rectangle(high, low, close, vol, vol_avg))
        res.update(detect_symmetrical_triangle(high, low, close))
        res.update(detect_rounding_bottom(close))
    except Exception:
        return {"v3_tags": []}

    tags = []
    if res.get("falling_wedge"):        tags.append("🔻 Falling Wedge Breakout")
    if res.get("rising_wedge"):         tags.append("🔺 Rising Wedge Breakdown")
    if res.get("bullish_pennant"):      tags.append("🎌 Bullish Pennant")
    if res.get("bearish_pennant"):      tags.append("🏴 Bearish Pennant")
    if res.get("bear_flag"):            tags.append("🐻 Bear Flag")
    if res.get("rectangle_breakout"):   tags.append("▭ Rectangle Breakout")
    if res.get("rounding_bottom"):      tags.append("🥣 Rounding Bottom")
    if res.get("symmetrical_triangle"): tags.append("📐 Symmetrical Triangle")
    res["v3_tags"] = tags
    return res


# ── Pattern strength registry ────────────────────────────────────────────────
# tier: 3 = Strong, 2 = Medium, 1 = Weak (context only)
# dir : +1 bullish, -1 bearish, 0 neutral/undecided
PATTERN_STRENGTH = {
    # Tier 1 / strong bullish
    "🛤️ Inverse H&S (Bottom)":     (3, +1),
    "📉 Double Bottom":             (3, +1),
    "☕ Cup & Handle Breakout":     (3, +1),
    "🚩 Bull Flag Breakout":        (3, +1),
    "📐 Ascending Triangle":        (3, +1),
    "🚀 Vol Breakout":              (3, +1),
    "🔻 Falling Wedge Breakout":    (3, +1),
    "🥣 Rounding Bottom":           (3, +1),
    # Strong bearish
    "🏔️ Head & Shoulders (Top)":   (3, -1),
    "📈 Double Top":                (3, -1),
    "📐 Descending Triangle":       (3, -1),
    "🔺 Rising Wedge Breakdown":    (3, -1),
    "🐻 Bear Flag":                 (3, -1),
    # Tier 2 / medium
    "🎌 Bullish Pennant":           (2, +1),
    "🏴 Bearish Pennant":           (2, -1),
    "▭ Rectangle Breakout":         (2, +1),
    "💰 Pocket Pivot":              (2, +1),
    "🎯 NR7 Inside Bar (coiled)":   (2, +1),
    "📐 Symmetrical Triangle":      (2, 0),
    "🟩 Bullish Engulfing":         (2, +1),
    "🟥 Bearish Engulfing":         (2, -1),
    "🌅 Morning Star":              (2, +1),
    "🌆 Evening Star":              (2, -1),
    "🪖 Three White Soldiers":      (2, +1),
    "🦅 Three Black Crows":         (2, -1),
    "🔆 Piercing Line":             (2, +1),
    # Tier 3 / weak — context only, never a standalone entry
    "🔨 Bullish Hammer":            (1, +1),
    "💫 Shooting Star":             (1, -1),
    "🟢 Bullish Harami":            (1, +1),
    "📦 Inside Bar":                (1, 0),
    "🤏 NR7 (narrow range)":        (1, 0),
    "〰️ Doji (Indecision)":        (1, 0),
}


def pattern_strength(tags, vol_ratio=None, trend=None, near_resistance=None):
    """Grade the pattern evidence on a row.

    Returns dict(label, score, tier, direction, best, confirmations).

    The grade is deliberately NOT just "which pattern" — the same pattern is
    worth far more with volume expansion and a matching trend behind it, which
    is exactly the point the user's own framework makes:
        Pattern = setup, Volume = confirmation, Trend = direction.
    """
    tags = [t for t in (tags or []) if t]
    if not tags:
        return {"label": "—", "score": 0, "tier": 0, "direction": 0,
                "best": None, "confirmations": []}

    known = [(t, PATTERN_STRENGTH[t]) for t in tags if t in PATTERN_STRENGTH]
    if not known:
        return {"label": "—", "score": 0, "tier": 0, "direction": 0,
                "best": None, "confirmations": []}

    # strongest pattern present drives the grade
    best_tag, (best_tier, best_dir) = max(known, key=lambda kv: kv[1][0])
    score = best_tier * 2                      # 6 / 4 / 2

    # stacking: a second independent pattern in the SAME direction adds weight
    same_dir = [t for t, (ti, d) in known
                if d == best_dir and d != 0 and t != best_tag]
    if same_dir:
        score += 1

    confirmations = []
    if vol_ratio and vol_ratio >= 1.5:
        score += 2; confirmations.append(f"volume {vol_ratio:.1f}x")
    elif vol_ratio and vol_ratio >= 1.2:
        score += 1; confirmations.append(f"volume {vol_ratio:.1f}x")

    if trend and best_dir != 0:
        up = "Uptrend" in str(trend)
        down = "Downtrend" in str(trend)
        if (best_dir > 0 and up) or (best_dir < 0 and down):
            score += 2; confirmations.append("trend aligned")
        elif (best_dir > 0 and down) or (best_dir < 0 and up):
            score -= 2; confirmations.append("⚠️ against trend")

    if near_resistance:
        score += 1; confirmations.append("at key level")

    if score >= 9:   label = "🔥 Very Strong"
    elif score >= 7: label = "🟢 Strong"
    elif score >= 4: label = "🟡 Medium"
    elif score >= 1: label = "🟠 Weak"
    else:            label = "🔴 Conflicting"

    return {"label": label, "score": int(score), "tier": best_tier,
            "direction": best_dir, "best": best_tag,
            "confirmations": confirmations}
