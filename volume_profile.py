"""
volume_profile.py — Volume Profile analytics (additive; does NOT modify signals.py).

Replaces the weak "support/resistance = 20-day rolling min/max" idea with REAL
volume-based levels: where trading actually happened, not just where price
briefly touched.

Core outputs (all computed from OHLCV, no extra data feed needed):

  POC  (Point of Control)  — the single price level with the MOST traded volume
                             over the lookback. This is the strongest magnet /
                             support-resistance level in the range.
  VAH / VAL (Value Area)   — the high/low bounds of the price band containing
                             ~70% of all traded volume. Breakouts ABOVE VAH or
                             breakdowns BELOW VAL are high-conviction moves.
  HVN (High Volume Nodes)  — price shelves where lots of volume sits → strong
                             support/resistance, price tends to stall here.
  LVN (Low Volume Nodes)   — price gaps with little volume → price moves FAST
                             through these → good breakout target zones.

Why it matters for "good entries + faster profit":
  - Entry: a breakout from a tight base that clears the POC/VAH on volume is a
    far higher-quality entry than "price > 20-day high" (which ignores whether
    anyone actually traded there).
  - Faster profit: the nearest LVN above entry is a natural fast-move target —
    price accelerates through low-volume gaps, so targets sitting just below the
    next HVN capture the quick move before it stalls.

Method: a histogram of volume across price bins. Since daily bars don't carry
intraday tick distribution, each bar's volume is spread across its High-Low
range (a standard, defensible approximation for daily Volume Profile).
"""

import numpy as np


def compute_volume_profile(high, low, close, volume, bins=50, value_area_pct=0.70):
    """Build a volume-by-price histogram and derive POC / Value Area / nodes.

    Args:
        high, low, close, volume: pandas Series (or array-likes) of equal length.
        bins: number of price buckets (50 is a good default for daily swing).
        value_area_pct: fraction of volume defining the Value Area (0.70 = std).

    Returns a dict (all prices rounded to 2dp), or None if not enough data:
        poc, vah, val, hvn (list), lvn (list), price_bins, volume_hist,
        poc_volume_share, in_value_area (is current close inside the VA?)
    """
    try:
        h = np.asarray(high, dtype=float)
        l = np.asarray(low, dtype=float)
        c = np.asarray(close, dtype=float)
        v = np.asarray(volume, dtype=float)
    except Exception:
        return None

    mask = ~(np.isnan(h) | np.isnan(l) | np.isnan(c) | np.isnan(v))
    h, l, c, v = h[mask], l[mask], c[mask], v[mask]
    if len(c) < 20:
        return None

    price_lo = float(np.min(l))
    price_hi = float(np.max(h))
    if price_hi <= price_lo:
        return None

    edges = np.linspace(price_lo, price_hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    vol_hist = np.zeros(bins, dtype=float)

    # Spread each bar's volume evenly across the price bins its range covers.
    bin_width = (price_hi - price_lo) / bins
    for i in range(len(c)):
        bar_lo, bar_hi, bar_v = l[i], h[i], v[i]
        if bar_hi <= bar_lo:
            # zero-range bar: dump all volume in the single bin containing close
            idx = min(int((c[i] - price_lo) / bin_width), bins - 1)
            idx = max(0, idx)
            vol_hist[idx] += bar_v
            continue
        lo_bin = max(0, int((bar_lo - price_lo) / bin_width))
        hi_bin = min(bins - 1, int((bar_hi - price_lo) / bin_width))
        span = hi_bin - lo_bin + 1
        if span <= 0:
            continue
        vol_hist[lo_bin:hi_bin + 1] += bar_v / span

    total_vol = vol_hist.sum()
    if total_vol <= 0:
        return None

    # ── POC: bin with the most volume ─────────────────────────────────────────
    poc_idx = int(np.argmax(vol_hist))
    poc = float(centers[poc_idx])
    poc_share = float(vol_hist[poc_idx] / total_vol)

    # ── Value Area: expand out from POC until ~70% of volume is captured ──────
    target = total_vol * value_area_pct
    captured = vol_hist[poc_idx]
    lo_i = hi_i = poc_idx
    while captured < target and (lo_i > 0 or hi_i < bins - 1):
        # look at the next bin on each side; take whichever has more volume
        left_v = vol_hist[lo_i - 1] if lo_i > 0 else -1
        right_v = vol_hist[hi_i + 1] if hi_i < bins - 1 else -1
        if right_v >= left_v:
            hi_i += 1
            captured += vol_hist[hi_i]
        else:
            lo_i -= 1
            captured += vol_hist[lo_i]
    val = float(centers[lo_i])
    vah = float(centers[hi_i])

    # ── HVN / LVN: local peaks & troughs of the volume histogram ─────────────
    avg_v = total_vol / bins
    hvn, lvn = [], []
    for i in range(1, bins - 1):
        if vol_hist[i] > vol_hist[i - 1] and vol_hist[i] > vol_hist[i + 1]:
            if vol_hist[i] > avg_v * 1.3:                      # meaningful shelf
                hvn.append(round(float(centers[i]), 2))
        if vol_hist[i] < vol_hist[i - 1] and vol_hist[i] < vol_hist[i + 1]:
            if vol_hist[i] < avg_v * 0.5:                      # genuine gap
                lvn.append(round(float(centers[i]), 2))

    cmp = float(c[-1])
    return {
        "poc": round(poc, 2),
        "vah": round(vah, 2),
        "val": round(val, 2),
        "poc_volume_share": round(poc_share, 3),
        "hvn": hvn[:6],
        "lvn": lvn[:6],
        "in_value_area": bool(val <= cmp <= vah),
        "above_value_area": bool(cmp > vah),
        "below_value_area": bool(cmp < val),
        "price_low": round(price_lo, 2),
        "price_high": round(price_hi, 2),
    }


def vp_support_resistance(vp, cmp):
    """From a volume profile, pick the nearest STRONG support (below) and
    resistance (above) using HVN + POC + Value-Area edges — a volume-backed
    replacement for rolling-min/max S/R. Returns (support, resistance)."""
    if not vp:
        return None, None
    levels = set()
    for key in ("poc", "vah", "val"):
        if vp.get(key):
            levels.add(vp[key])
    for lv in vp.get("hvn", []):
        levels.add(lv)
    below = [x for x in levels if x < cmp]
    above = [x for x in levels if x > cmp]
    support = max(below) if below else None
    resistance = min(above) if above else None
    return support, resistance


def vp_fast_move_target(vp, entry):
    """The nearest LVN above entry = the level price tends to reach QUICKLY
    (low-volume gap → little resistance). Good 'fast profit' target. Falls back
    to VAH, then price_high. Returns a price or None."""
    if not vp:
        return None
    lvns_above = sorted([x for x in vp.get("lvn", []) if x > entry])
    if lvns_above:
        return lvns_above[0]
    if vp.get("vah") and vp["vah"] > entry:
        return vp["vah"]
    return vp.get("price_high")


def vp_entry_quality(vp, cmp, breakout_level=None):
    """Score how good the CURRENT location is for a breakout entry, 0-100, using
    volume-profile context. Higher = cleaner room to run.

    Rationale:
      - Breaking ABOVE the value area / POC on a fresh move = institutional
        acceptance of higher prices → strong.
      - Sitting just under a big HVN = overhead supply → weak (will stall).
      - Clear LVN space above = fast-move room → strong.
    """
    if not vp or not cmp:
        return None
    score = 50.0
    poc = vp.get("poc")

    # Above value area = acceptance of higher prices
    if vp.get("above_value_area"):
        score += 20
    elif vp.get("in_value_area"):
        score += 5
    else:
        score -= 10   # below value area = still in / under supply

    # Clear of the POC (the biggest magnet) to the upside
    if poc and cmp > poc:
        score += 10

    # Overhead supply check: is there a big HVN close above? (bad → stall)
    hvn_above = [x for x in vp.get("hvn", []) if x > cmp]
    if hvn_above:
        nearest = min(hvn_above)
        gap_pct = (nearest - cmp) / cmp * 100
        if gap_pct < 2:
            score -= 15      # wall right overhead
        elif gap_pct < 5:
            score -= 5
        else:
            score += 5       # room before the next shelf

    # Fast-move room: is there an LVN gap above to accelerate into?
    lvn_above = [x for x in vp.get("lvn", []) if x > cmp]
    if lvn_above:
        score += 10

    return int(round(max(0, min(100, score))))


if __name__ == "__main__":
    # Self-test on synthetic data with a KNOWN high-volume shelf
    np.random.seed(1)
    n = 120
    # Build price that spends lots of time (volume) around 100, then breaks to 110
    base = np.concatenate([
        np.random.normal(100, 1.5, 80),    # heavy accumulation near 100 → POC here
        np.linspace(101, 110, 40),          # breakout leg up
    ])
    high = base + np.random.uniform(0.3, 1.0, n)
    low  = base - np.random.uniform(0.3, 1.0, n)
    close = base + np.random.uniform(-0.3, 0.3, n)
    vol = np.concatenate([
        np.random.uniform(2e6, 3e6, 80),    # high volume in the base
        np.random.uniform(0.5e6, 1e6, 40),  # lighter volume on the breakout
    ])
    vp = compute_volume_profile(high, low, close, vol)
    print("Volume Profile:")
    for k, val in vp.items():
        print(f"  {k}: {val}")
    print(f"\nPOC should be near 100 (the accumulation shelf): POC={vp['poc']}")
    s, r = vp_support_resistance(vp, cmp=float(close[-1]))
    print(f"Volume-based S/R at cmp {close[-1]:.1f}: support={s} resistance={r}")
    print(f"Fast-move target from entry {close[-1]:.1f}: {vp_fast_move_target(vp, close[-1])}")
    print(f"Entry quality score: {vp_entry_quality(vp, float(close[-1]))}")
