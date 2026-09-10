"""
nse_data.py — delivery % and bulk/block deals from NSE.

WHY THIS MATTERS
----------------
Delivery percentage is the most honest India-specific proxy for accumulation
that retail actually has access to. A breakout on 60% delivery means buyers
took stock home; the same breakout on 15% delivery is intraday churn that
usually gives the move back. Bulk and block deals are a genuine institutional
footprint — a named counterparty crossing size.

This is what people usually mean when they ask about "operator activity". The
difference is that these are PUBLISHED, dated and falsifiable, whereas
"operator" as commonly used is an unfalsifiable label applied after the move.

⚠️ HONEST WARNING ABOUT RELIABILITY
-----------------------------------
NSE blocks unauthenticated requests and is known to reject cloud/data-centre
IPs. From the environment this module was written in, both nseindia.com and
archives.nseindia.com returned HTTP 403, so IT COULD NOT BE VERIFIED END TO END
against the live site.

That is why every function here fails soft and returns None rather than
guessing, and why `health_check()` exists: run it once from wherever you deploy
and you will know in seconds whether this data is available to you at all,
instead of silently trusting numbers that never arrived.

If it does not work on Streamlit Cloud, that is the same class of problem as the
Angel One API being unreachable there — a host/network limitation, not a bug in
this file. It has a much better chance from a local machine or a normal VPS.
"""

import time
import datetime as _dt

import requests

_BASE = "https://www.nseindia.com"
_UA = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": _BASE + "/get-quotes/equity",
}

_SESSION = None
_SESSION_TS = 0
_SESSION_TTL = 600          # re-bootstrap cookies every 10 min

_CACHE = {}
_CACHE_TTL = 3600           # delivery data is end-of-day; an hour is plenty


def _clean(symbol):
    s = str(symbol).upper().strip()
    for sfx in (".NS", ".BO", ".NSE", ".BSE"):
        if s.endswith(sfx):
            s = s[: -len(sfx)]
    return s


def _session():
    """NSE requires cookies from a normal page visit before its JSON APIs will
    answer. Hitting the API directly returns 401/403 even with a browser UA."""
    global _SESSION, _SESSION_TS
    now = time.time()
    if _SESSION is not None and (now - _SESSION_TS) < _SESSION_TTL:
        return _SESSION
    try:
        s = requests.Session()
        s.headers.update(_UA)
        s.get(_BASE, timeout=8)                       # sets the initial cookies
        s.get(_BASE + "/get-quotes/equity?symbol=RELIANCE", timeout=8)
        _SESSION, _SESSION_TS = s, now
        return s
    except Exception:
        _SESSION, _SESSION_TS = None, 0
        return None


def delivery_stats(symbol, timeout=8):
    """Delivery data for one symbol.

    Returns {deliverable_pct, traded_qty, delivered_qty, date} or None when the
    data is genuinely unavailable. None means UNKNOWN — never treat it as zero,
    and never penalise a stock for it.
    """
    sym = _clean(symbol)
    now = time.time()
    hit = _CACHE.get(("dp", sym))
    if hit and (now - hit[0]) < _CACHE_TTL:
        return hit[1]

    out = None
    try:
        s = _session()
        if s is not None:
            r = s.get(f"{_BASE}/api/quote-equity?symbol={sym}&section=trade_info",
                      timeout=timeout)
            if r.ok:
                j = r.json()
                dp = (j.get("securityWiseDP") or {})
                pct = dp.get("deliveryToTradedQuantity")
                if pct is not None:
                    out = {
                        "deliverable_pct": round(float(pct), 2),
                        "traded_qty": dp.get("quantityTraded"),
                        "delivered_qty": dp.get("deliveryQuantity"),
                        "date": dp.get("secWiseDelPosDate"),
                    }
    except Exception:
        out = None

    _CACHE[("dp", sym)] = (now, out)
    return out


def delivery_bulk(symbols, max_symbols=40, deadline_s=15):
    """Delivery stats for a SMALL list, threaded with a hard deadline.

    Bounded on purpose. Fetching this for a 2000-symbol universe would be both
    abusive to NSE and the exact pattern that hung the scanner when Angel LTP
    was called per symbol.
    """
    import concurrent.futures as _cf
    out = {}
    syms = list(dict.fromkeys([_clean(s) for s in symbols if s]))[:max_symbols]
    if not syms:
        return out
    ex = _cf.ThreadPoolExecutor(max_workers=5)
    try:
        futs = {ex.submit(delivery_stats, s): s for s in syms}
        try:
            for f in _cf.as_completed(futs, timeout=deadline_s):
                sym = futs[f]
                try:
                    out[sym] = f.result()
                except Exception:
                    out[sym] = None
        except _cf.TimeoutError:
            pass
    except Exception:
        pass
    finally:
        ex.shutdown(wait=False)
    return out


def bulk_deals(days_back=7, timeout=10):
    """Recent bulk deals across the market.

    Returns a list of {date, symbol, client, buy_sell, qty, price} or [] when
    unavailable. Bulk deals are >0.5% of listed shares in a single session, so
    they are a real institutional/large-player footprint rather than a rumour.
    """
    try:
        s = _session()
        if s is None:
            return []
        to_d = _dt.date.today()
        from_d = to_d - _dt.timedelta(days=days_back)
        url = (f"{_BASE}/api/historical/bulk-deals"
               f"?from={from_d.strftime('%d-%m-%Y')}&to={to_d.strftime('%d-%m-%Y')}")
        r = s.get(url, timeout=timeout)
        if not r.ok:
            return []
        rows = (r.json() or {}).get("data", []) or []
        out = []
        for d in rows:
            out.append({
                "date": d.get("mTIMESTAMP") or d.get("BD_DT_DATE"),
                "symbol": d.get("BD_SYMBOL"),
                "client": d.get("BD_CLIENT_NAME"),
                "buy_sell": d.get("BD_BUY_SELL"),
                "qty": d.get("BD_QTY_TRD"),
                "price": d.get("BD_TP_WATP"),
            })
        return out
    except Exception:
        return []


def bulk_deal_map(days_back=7):
    """{SYMBOL: {buys, sells, last_date, clients}} — quick lookup for a scanner
    row so you can see 'this name had 2 institutional buys this week'."""
    deals = bulk_deals(days_back)
    m = {}
    for d in deals:
        sym = _clean(d.get("symbol") or "")
        if not sym:
            continue
        e = m.setdefault(sym, {"buys": 0, "sells": 0, "last_date": None,
                               "clients": []})
        bs = str(d.get("buy_sell") or "").upper()
        if bs.startswith("B"):
            e["buys"] += 1
        elif bs.startswith("S"):
            e["sells"] += 1
        e["last_date"] = d.get("date") or e["last_date"]
        c = d.get("client")
        if c and c not in e["clients"]:
            e["clients"].append(c)
    return m


def delivery_note(stats):
    """One-line read of a delivery figure, or '' when unknown.

    Bands are the commonly used retail reference points, not precise science —
    the useful signal is the contrast between a high and a low reading on a
    breakout day, not the exact number.
    """
    if not stats or stats.get("deliverable_pct") is None:
        return ""
    p = stats["deliverable_pct"]
    if p >= 60:
        return f"📦 Delivery {p:.0f}% — strong, buyers took stock home"
    if p >= 40:
        return f"📦 Delivery {p:.0f}% — healthy"
    if p >= 25:
        return f"📦 Delivery {p:.0f}% — average"
    return f"⚠️ Delivery {p:.0f}% — mostly intraday churn, weak conviction"


def health_check():
    """Is NSE reachable from THIS host? Run once after deploying.

    Returns {session, delivery, bulk_deals, note}. This exists because the
    module could not be verified against the live site when it was written, and
    silently-missing data is worse than data you know you don't have.
    """
    res = {"session": False, "delivery": False, "bulk_deals": False, "note": ""}
    s = _session()
    res["session"] = s is not None
    if not res["session"]:
        res["note"] = ("Could not establish an NSE session - this host is very "
                       "likely blocked (NSE rejects many cloud/data-centre IPs). "
                       "Delivery % and bulk deals will be unavailable here.")
        return res
    d = delivery_stats("RELIANCE")
    res["delivery"] = bool(d and d.get("deliverable_pct") is not None)
    res["bulk_deals"] = len(bulk_deals(3)) > 0
    if res["delivery"] and res["bulk_deals"]:
        res["note"] = "NSE data is fully available from this host."
    elif res["delivery"]:
        res["note"] = "Delivery % works; bulk-deals endpoint did not respond."
    else:
        res["note"] = ("Session established but the data endpoints refused. "
                       "NSE is likely rate-limiting or blocking this host.")
    return res


if __name__ == "__main__":
    print("NSE reachability:", health_check())
