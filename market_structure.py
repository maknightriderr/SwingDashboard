"""
market_structure.py — the foundation layer under SMC / price action.

The SMC module already finds order blocks, fair-value gaps and liquidity pools.
Those are all *entry* tools, and in SMC methodology they are only meant to be
taken IN THE DIRECTION OF STRUCTURE. Without a structure read, an order block
is just a level with no idea whether the trend supports trading it — which is
the same failure as taking a chart pattern without checking the trend.

This module supplies that missing read:

  SWING CLASSIFICATION  Label each confirmed swing high/low as HH/LH and HL/LL.
                        Bullish structure = higher highs AND higher lows.

  BOS (Break of Structure)
                        Price closes beyond the most recent swing high (bullish
                        BOS) or swing low (bearish BOS) IN THE DIRECTION OF the
                        existing trend. BOS = continuation, and it is what
                        confirms a trend is still intact.

  CHoCH (Change of Character)
                        The first break AGAINST the prevailing structure — e.g.
                        an uptrend making higher highs suddenly closes below its
                        last higher low. CHoCH is an early warning that the
                        trend may be turning, and it usually fires well before a
                        trailing stop does. That makes it far more useful on a
                        HELD position than on a fresh entry.

Everything is computed from confirmed swing points (fractals with two bars
either side), so today's bar can never invent a swing that isn't there yet —
the same discipline used for the VCP pivot fix.
"""

import numpy as np


def _swings(high, low, left=2, right=2, min_gap=3):
    """Confirmed swing highs/lows as (index, price) lists.

    A swing needs `right` bars AFTER it, so the most recent bars can never be
    swings — deliberately. A "swing high" that hasn't been confirmed yet is just
    today's price, and treating it as structure is how you end up chasing.
    """
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    n = len(h)
    highs, lows = [], []
    for i in range(left, n - right):
        window_h = h[i - left:i + right + 1]
        window_l = l[i - left:i + right + 1]
        if h[i] == window_h.max() and (not highs or i - highs[-1][0] >= min_gap):
            highs.append((i, float(h[i])))
        if l[i] == window_l.min() and (not lows or i - lows[-1][0] >= min_gap):
            lows.append((i, float(l[i])))
    return highs, lows


def _label_swings(highs, lows):
    """Tag each swing as HH/LH and HL/LL relative to the previous one."""
    h_labels, l_labels = [], []
    for k, (i, p) in enumerate(highs):
        if k == 0:
            h_labels.append((i, p, "H"))
        else:
            h_labels.append((i, p, "HH" if p > highs[k - 1][1] else "LH"))
    for k, (i, p) in enumerate(lows):
        if k == 0:
            l_labels.append((i, p, "L"))
        else:
            l_labels.append((i, p, "HL" if p > lows[k - 1][1] else "LL"))
    return h_labels, l_labels


def analyse_structure(high, low, close, lookback=120):
    """Full market-structure read.

    Returns:
      structure      : 'Bullish' | 'Bearish' | 'Ranging' | 'Unknown'
      last_swing_high / last_swing_low : the levels that define the current
                       structure — a close beyond them is what triggers BOS/CHoCH
      bos            : 'Bullish' | 'Bearish' | None   (continuation)
      choch          : 'Bullish' | 'Bearish' | None   (early reversal warning)
      bos_level      : the level that was broken
      swing_labels   : recent HH/HL/LH/LL sequence, for display
      detail         : short human explanation
    """
    out = {"structure": "Unknown", "last_swing_high": None, "last_swing_low": None,
           "bos": None, "choch": None, "bos_level": None,
           "swing_labels": [], "detail": ""}

    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    if len(c) < 30:
        out["detail"] = "Not enough history for a structure read"
        return out

    h, l, c = h[-lookback:], l[-lookback:], c[-lookback:]
    highs, lows = _swings(h, l)
    if len(highs) < 2 or len(lows) < 2:
        out["detail"] = "Too few confirmed swings"
        return out

    h_lab, l_lab = _label_swings(highs, lows)
    out["last_swing_high"] = round(highs[-1][1], 2)
    out["last_swing_low"] = round(lows[-1][1], 2)

    # Interleave the last few swings in time order for a readable sequence
    recent = sorted(h_lab[-3:] + l_lab[-3:], key=lambda x: x[0])
    out["swing_labels"] = [t for _, _, t in recent]

    # ── Prevailing structure ────────────────────────────────────────────────
    last_h_type = h_lab[-1][2]
    last_l_type = l_lab[-1][2]
    if last_h_type == "HH" and last_l_type == "HL":
        structure = "Bullish"
    elif last_h_type == "LH" and last_l_type == "LL":
        structure = "Bearish"
    else:
        structure = "Ranging"
    out["structure"] = structure

    cmp_ = float(c[-1])
    swing_high = highs[-1][1]
    swing_low = lows[-1][1]

    # ── BOS vs CHoCH ────────────────────────────────────────────────────────
    # Same event, opposite meaning depending on the structure it happens in:
    #   break WITH the structure  -> BOS   (continuation, healthy)
    #   break AGAINST it          -> CHoCH (character change, warning)
    broke_up = cmp_ > swing_high
    broke_down = cmp_ < swing_low

    if structure == "Bullish":
        if broke_up:
            out["bos"] = "Bullish"; out["bos_level"] = round(swing_high, 2)
            out["detail"] = (f"Bullish BOS - closed above the last swing high "
                             f"{swing_high:.2f}; uptrend structure intact")
        elif broke_down:
            out["choch"] = "Bearish"; out["bos_level"] = round(swing_low, 2)
            out["detail"] = (f"Bearish CHoCH - an uptrend just closed below its "
                             f"last higher low {swing_low:.2f}; first sign the "
                             f"trend may be turning")
        else:
            out["detail"] = (f"Bullish structure (HH/HL) holding between "
                             f"{swing_low:.2f} and {swing_high:.2f}")
    elif structure == "Bearish":
        if broke_down:
            out["bos"] = "Bearish"; out["bos_level"] = round(swing_low, 2)
            out["detail"] = (f"Bearish BOS - closed below the last swing low "
                             f"{swing_low:.2f}; downtrend intact")
        elif broke_up:
            out["choch"] = "Bullish"; out["bos_level"] = round(swing_high, 2)
            out["detail"] = (f"Bullish CHoCH - a downtrend just closed above its "
                             f"last lower high {swing_high:.2f}; possible bottom "
                             f"forming")
        else:
            out["detail"] = (f"Bearish structure (LH/LL) between "
                             f"{swing_low:.2f} and {swing_high:.2f}")
    else:
        if broke_up:
            out["bos"] = "Bullish"; out["bos_level"] = round(swing_high, 2)
            out["detail"] = (f"Range break UP through {swing_high:.2f}")
        elif broke_down:
            out["bos"] = "Bearish"; out["bos_level"] = round(swing_low, 2)
            out["detail"] = (f"Range break DOWN through {swing_low:.2f}")
        else:
            out["detail"] = (f"Ranging between {swing_low:.2f} and "
                             f"{swing_high:.2f} - no clear structure")

    return out


def structure_tags(res):
    """Short display tags for the scanner's Patterns column."""
    tags = []
    if res.get("bos") == "Bullish":   tags.append("📶 Bullish BOS")
    elif res.get("bos") == "Bearish": tags.append("📉 Bearish BOS")
    if res.get("choch") == "Bearish": tags.append("⚡ Bearish CHoCH")
    elif res.get("choch") == "Bullish": tags.append("⚡ Bullish CHoCH")
    return tags


def structure_alignment(structure, direction):
    """Does a trade direction agree with the prevailing structure?

    direction: +1 long, -1 short, 0 neutral
    Returns (aligned: bool|None, note: str). None = no opinion (ranging/unknown).
    """
    if not structure or structure in ("Unknown", "Ranging") or direction == 0:
        return None, ""
    if direction > 0:
        if structure == "Bullish":
            return True, "structure aligned (HH/HL)"
        return False, "⚠️ long against bearish structure (LH/LL)"
    if structure == "Bearish":
        return True, "structure aligned (LH/LL)"
    return False, "⚠️ short against bullish structure (HH/HL)"
