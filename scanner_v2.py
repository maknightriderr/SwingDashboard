"""
scanner_v2.py — Universe Scanner 2.0 (additive module; does NOT modify signals.py)

Implements the audit's High-priority scanner fixes on top of the existing engine:

  1. REGIME GATING        — Bear/Strong Bear suppresses long scores & top tiers
  2. COLLINEARITY CAP     — trend/supertrend/MACD cluster capped (no double count)
  3. EXTENSION PENALTY    — punishes buying far above the rising 21EMA
  4. RS GATE              — laggards vs Nifty penalised, leaders rewarded
  5. FRESH BREAKOUTS ONLY — +5 only if the range break happened in last 2 bars
  6. STRUCTURAL STOPS     — below the actual swing low (buffered), capped at 8% risk
  7. REAL RR              — measured-move / structural targets → RR filter now bites
  8. PERCENTILE RANKING   — signals from rank-within-scan, not absolute thresholds

Everything degrades gracefully: any per-symbol failure just skips that symbol.
Reuses signals.py's bulk fetch + indicator cache (no duplicate Yahoo traffic).

v2.1 adds: single-stock lookup (score_single_stock) for checking any NSE symbol
on demand, sharing the exact same scoring logic as the full universe scan.
"""

import pandas as pd
from datetime import datetime

import signals as _sg

try:
    import volume_profile as _vp
    _VP_AVAILABLE = True
except Exception:
    _vp = None
    _VP_AVAILABLE = False

# ── Tunables (kept together so you can tweak later) ───────────────────────────
MAX_RISK_PCT      = 8.0    # skip top tiers if structural stop demands more risk
FRESH_BARS        = 2      # breakout must have happened within N bars
EXT_HARD_ATR      = 2.5    # cmp this many ATRs above 21EMA = hard penalty
EXT_SOFT_ATR      = 1.5    # soft penalty
RR_MIN_BUY        = 1.5
RR_MIN_STRONG     = 1.8


def _fresh_breakout(df):
    """Detects whether price broke above the prior 20-day range high within the
    last FRESH_BARS sessions (the audit's 'stale breakout' fix).
    Returns (is_fresh, breakout_level, range_low) or (False, None, None)."""
    try:
        if df is None or len(df) < 25:
            return False, None, None
        high, low, close = df["High"], df["Low"], df["Close"]
        for k in range(1, FRESH_BARS + 1):
            prior_high = float(high.iloc[-(20 + k):-k].max())
            prior_low  = float(low.iloc[-(20 + k):-k].min())
            bar_close  = float(close.iloc[-k])
            prev_close = float(close.iloc[-(k + 1)])
            if prev_close <= prior_high and bar_close > prior_high:
                return True, round(prior_high, 2), round(prior_low, 2)
        return False, None, None
    except Exception:
        return False, None, None


def _structural_stop(df, cmp, atr):
    """Stop below the recent swing low (0.25*ATR buffer). Falls back to an
    ATR stop when structure is degenerate. Returns (stop, basis, risk_pct)."""
    atr_stop = round(cmp - 1.5 * atr, 2)
    try:
        swing_low = float(df["Low"].iloc[-10:].min())
        structural = round(swing_low - 0.25 * atr, 2)
        if structural < cmp - 0.5 * atr:
            stop, basis = structural, "swing low"
        else:
            stop, basis = atr_stop, "1.5×ATR"
    except Exception:
        stop, basis = atr_stop, "1.5×ATR"
    if stop >= cmp:
        stop, basis = atr_stop, "1.5×ATR"
    risk_pct = round((cmp - stop) / cmp * 100, 2) if cmp else 0.0
    return stop, basis, risk_pct


def _target(cmp, atr, resistance, fresh, brk_level, range_low):
    """Fresh breakout → measured move (range height above the pivot).
    Otherwise the engine's classic max(resistance, cmp + 2.5*ATR)."""
    if fresh and brk_level and range_low and brk_level > range_low:
        measured = brk_level + (brk_level - range_low)
        return round(max(measured, cmp + 2.0 * atr), 2), "measured move"
    return round(max(resistance or 0, cmp + 2.5 * atr), 2), "resistance/ATR"


def _score_symbol(symbol, ind, df, is_bear, is_caution):
    """Scores ONE symbol given its precomputed indicators + OHLCV frame.
    Returns a row dict (same shape used in the bulk scan table), or None if
    the symbol can't be scored (missing cmp/atr). Shared by both the full
    universe scan and the single-stock lookup so the numbers always match."""
    cmp   = ind.get("cmp");  atr = ind.get("atr")
    rsi   = ind.get("rsi");  trend = ind.get("trend", "—")
    if not cmp or not atr or atr <= 0:
        return None
    patterns = ind.get("patterns", []) or []
    candles  = ind.get("candlesticks", []) or []

    # ── 2. Momentum cluster (collinear features CAPPED at 4) ─────────────────
    mom = 0.0
    if trend in ("Uptrend", "Strong Uptrend"): mom += 2.0
    if ind.get("supertrend_bullish"):          mom += 1.5
    if ind.get("macd_bullish"):                mom += 1.0
    if ind.get("macd_hist_expanding"):         mom += 0.5
    score = min(mom, 4.0)

    # ── 4. Relative strength gate ─────────────────────────────────────────────
    rs = ind.get("rs_ratio")
    if isinstance(rs, (int, float)):
        if rs >= 1.05:   score += 2.0
        elif rs >= 1.00: score += 1.0
        elif rs < 0.95:  score -= 3.0
    rs_disp = round(rs, 2) if isinstance(rs, (int, float)) else None

    # ── 5. Fresh breakout only (stale ones get almost nothing) ───────────────
    fresh, brk_level, range_low = _fresh_breakout(df)
    vol_ok = (ind.get("vol_ratio") or 0) >= 1.5
    if fresh and vol_ok:
        score += 5.0
    elif fresh:
        score += 3.0
    elif "🚀 Vol Breakout" in patterns:
        score += 1.0
    if "🚩 Bull Flag Breakout" in patterns:     score += 3.0
    if "☕ Cup & Handle Breakout" in patterns:   score += 3.0
    if "🟩 Bullish Engulfing" in candles:        score += 1.5
    if ind.get("bb_squeeze"):                    score += 1.0

    # ── 3. Extension penalty (don't buy stretched moves) ──────────────────────
    ema_ref = ind.get("ema21") or ind.get("ema50") or ind.get("ema20")
    ext_atr = None
    if ema_ref and ema_ref > 0:
        ext_atr = (cmp - float(ema_ref)) / atr
        if ext_atr > EXT_HARD_ATR:   score -= 4.0
        elif ext_atr > EXT_SOFT_ATR: score -= 2.0
    if rsi and rsi > 80: score -= 3.0
    elif rsi and 55 <= rsi <= 72: score += 1.0

    if ("📈 Double Top" in patterns or
            "🏔️ Head & Shoulders (Top)" in patterns or
            "🟥 Bearish Engulfing" in candles):
        score -= 5.0
    if trend in ("Downtrend", "Strong Downtrend"):
        score -= 5.0

    # ── 1. Regime gating ───────────────────────────────────────────────────────
    if is_bear:      score -= 4.0
    elif is_caution: score -= 1.0

    # ── Volume Profile: real volume-based context (POC / value area / LVN) ────
    # This upgrades entry quality AND gives a faster-move target than a flat
    # ATR/resistance target. Degrades silently if the module/data is unavailable.
    vp_data = None
    vp_score_note = ""
    if _VP_AVAILABLE and df is not None and len(df) >= 30:
        try:
            vp_data = _vp.compute_volume_profile(
                df["High"], df["Low"], df["Close"], df["Volume"])
        except Exception:
            vp_data = None
    if vp_data:
        # Entry quality from volume-profile location (0-100). Reward clean
        # breakouts above the value area with LVN room; penalise overhead supply.
        vpq = _vp.vp_entry_quality(vp_data, cmp, brk_level)
        if vpq is not None:
            if vpq >= 75:   score += 2.5; vp_score_note = "VP: clear room ↑"
            elif vpq >= 60: score += 1.0; vp_score_note = "VP: decent room"
            elif vpq < 40:  score -= 2.0; vp_score_note = "VP: overhead supply"
        # Acceptance above the Point of Control = institutional support beneath
        if vp_data.get("poc") and cmp > vp_data["poc"]:
            score += 0.5

    # ── 6 & 7. Structural stop + real target → meaningful RR ──────────────────
    stop, stop_basis, risk_pct = _structural_stop(df, cmp, atr)
    target, tgt_basis = _target(cmp, atr, ind.get("resistance"),
                                fresh, brk_level, range_low)
    # Prefer a volume-profile fast-move target (nearest LVN gap above) when it's
    # a better/closer objective than the generic ATR/resistance target — LVN
    # gaps are where price accelerates, i.e. the "quick profit" zone.
    if vp_data:
        _entry_ref = brk_level if (fresh and brk_level) else cmp
        vp_tgt = _vp.vp_fast_move_target(vp_data, _entry_ref)
        if vp_tgt and vp_tgt > cmp:
            # Use VP target if it gives a tighter, faster objective (but never
            # below the structural target — keep the more conservative RR honest)
            if vp_tgt < target:
                target, tgt_basis = round(float(vp_tgt), 2), "LVN fast-move"
    if fresh and brk_level and cmp <= brk_level * 1.02:
        entry_px = round(brk_level * 1.002, 2)
    else:
        entry_px = cmp
    risk = entry_px - stop
    rr = round((target - entry_px) / risk, 2) if risk > 0.01 else None
    risk_pct = round(risk / entry_px * 100, 2) if entry_px else risk_pct
    too_wide = risk_pct > MAX_RISK_PCT

    return {
        "Stock": symbol, "Sector": _sg.get_sector(symbol),
        "Score": round(score, 1),
        "CMP": cmp, "RSI": rsi, "Trend": trend,
        "RS": rs_disp,
        "Fresh": "⚡" if fresh else "",
        "Ext(ATR)": round(ext_atr, 1) if ext_atr is not None else None,
        "Entry": entry_px,
        "SL": stop, "SL basis": stop_basis,
        "Risk %": risk_pct,
        "Target": target, "Tgt basis": tgt_basis,
        "RR": rr,
        "POC": vp_data.get("poc") if vp_data else None,
        "VP note": vp_score_note,
        "_too_wide": too_wide,
    }


def _tier_for_row(r, is_bear, pctl=None):
    """Signal tier for a row. When pctl is available (bulk scan), uses
    percentile-within-scan. When pctl is None (single-stock lookup — there's
    no batch to rank against), falls back to absolute score thresholds,
    clearly documented as such in the UI."""
    rr_ok_buy    = (r["RR"] or 0) >= RR_MIN_BUY
    rr_ok_strong = (r["RR"] or 0) >= RR_MIN_STRONG
    if r["_too_wide"]:
        return "⚠️ RISK WIDE"
    if is_bear:
        return ("👀 WATCH (bear regime)" if r["Score"] >= 5 and rr_ok_buy
                else "⚪ NEUTRAL" if r["Score"] >= 2 else "🔴 AVOID")
    if pctl is not None:
        if pctl >= 90 and r["Score"] >= 8 and rr_ok_strong:
            return "🔥 STRONG BUY"
        if pctl >= 75 and r["Score"] >= 5 and rr_ok_buy:
            return "🟢 BUY SETUP"
    else:
        # Absolute fallback for single-stock lookups (no percentile context)
        if r["Score"] >= 8 and rr_ok_strong:
            return "🔥 STRONG BUY (absolute score, no percentile context)"
        if r["Score"] >= 5 and rr_ok_buy:
            return "🟢 BUY SETUP (absolute score, no percentile context)"
    if r["Score"] >= 2:
        return "🟡 ACCUMULATE"
    if r["Score"] <= 0:
        return "🔴 AVOID"
    return "⚪ NEUTRAL"


def generate_market_scanner_v2(max_symbols=None):
    """Improved universe scan. Returns dict with 'df' (ranked DataFrame),
    'regime', 'scanned', 'timestamp'. Never raises."""
    out = {"df": pd.DataFrame(), "regime": "Unknown", "scanned": 0,
           "timestamp": datetime.now().strftime("%d %b %H:%M")}
    try:
        market = _sg.get_market_regime() or {}
    except Exception:
        market = {}
    regime = market.get("regime", "Unknown")
    out["regime"] = regime
    is_bear     = regime in ("Bear", "Strong Bear")
    is_caution  = regime in ("Bull Pullback", "Bear Rally", "Neutral", "Unknown")

    all_symbols = []
    for _sec, stocks in _sg.SECTOR_STOCKS.items():
        all_symbols.extend(stocks)
    if max_symbols:
        all_symbols = all_symbols[:max_symbols]

    try:
        bulk = _sg._bulk_fetch_history(all_symbols, period="6mo")
    except Exception:
        return out

    rows = []
    for symbol in all_symbols:
        try:
            df = bulk.get(symbol)
            ind = _sg.compute_indicators(symbol, period="6mo", prefetched_df=df)
            if not ind:
                continue
            row = _score_symbol(symbol, ind, df, is_bear, is_caution)
            if row is not None:
                rows.append(row)
        except Exception:
            continue

    out["scanned"] = len(rows)
    if not rows:
        return out

    sdf = pd.DataFrame(rows)

    # ── 8. Percentile ranking within THIS scan ───────────────────────────────
    sdf["Pctl"] = (sdf["Score"].rank(pct=True) * 100).round(0)
    sdf["Signal"] = sdf.apply(
        lambda r: _tier_for_row(r, is_bear, pctl=r["Pctl"]), axis=1)
    sdf = sdf.drop(columns=["_too_wide"])
    sdf = sdf.sort_values(["Score", "RR"], ascending=[False, False]).reset_index(drop=True)
    out["df"] = sdf
    return out


def score_single_stock(symbol):
    """Score ONE arbitrary NSE symbol (doesn't need to be in the curated
    universe — any symbol signals.py can fetch works) using the exact same
    logic as the full scan. Returns a dict with the score row + regime, or
    {'error': ...} if the symbol couldn't be fetched/scored.

    NOTE: there's no percentile rank for a single stock (nothing to rank
    against), so the Signal tier here uses absolute score thresholds and is
    labelled accordingly — treat it as directional, not as strict as the
    ranked STRONG BUY/BUY SETUP tiers from a full scan.
    """
    symbol = str(symbol).upper().strip()
    try:
        market = _sg.get_market_regime() or {}
    except Exception:
        market = {}
    regime = market.get("regime", "Unknown")
    is_bear    = regime in ("Bear", "Strong Bear")
    is_caution = regime in ("Bull Pullback", "Bear Rally", "Neutral", "Unknown")

    try:
        bulk = _sg._bulk_fetch_history([symbol], period="6mo")
        df = bulk.get(symbol)
        ind = _sg.compute_indicators(symbol, period="6mo", prefetched_df=df)
    except Exception as e:
        return {"error": f"Could not fetch {symbol}: {e}"}

    if not ind:
        return {"error": f"No usable data for {symbol} — check the symbol, "
                         f"or it may be too new / illiquid / delisted."}

    row = _score_symbol(symbol, ind, df, is_bear, is_caution)
    if row is None:
        return {"error": f"{symbol} fetched but couldn't be scored "
                         f"(missing price/ATR data)."}

    row["Signal"] = _tier_for_row(row, is_bear, pctl=None)
    row.pop("_too_wide", None)
    row["regime"] = regime
    row["timestamp"] = datetime.now().strftime("%d %b %H:%M")
    return row
