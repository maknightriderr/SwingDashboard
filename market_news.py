"""
market_news.py — Market-wide Indian trading news (additive module).

Replaces the holdings-only news with a proper multi-source feed covering the
whole market: indices, stocks, results, IPOs, economy/RBI policy, commodities,
currency and global cues.

DESIGN NOTES
------------
* Sources are PRIMARY publishers (Business Standard, Livemint, Economic Times,
  Moneycontrol), not just a Google News aggregator. Google News RSS is kept only
  as a per-symbol fallback for stock-specific lookups.
* Business Standard's URLs below were taken from its official RSS listing page,
  so they are known-good. The others are the publishers' standard feed paths;
  if any one of them changes or blocks the request, that feed is simply SKIPPED
  (never crashes the page) and `feed_health()` will show it as failing.
* Every feed is fetched with a short timeout and the whole set is fetched in
  parallel, so a slow source can't stall the dashboard.
* Results are de-duplicated across sources (the same story is often carried by
  several outlets) and sorted newest-first.

USAGE
-----
    import market_news as mn
    items = mn.fetch_market_news()                     # everything, newest first
    items = mn.fetch_market_news(categories=["Markets", "Results"])
    items = mn.fetch_stock_news("RELIANCE")            # one symbol
    health = mn.feed_health()                          # which feeds are alive

Each item: {title, link, source, category, published (datetime|None), age}
"""

import re
import time
import html
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime

import requests

_IST = timezone(timedelta(hours=5, minutes=30))
_UA = {"User-Agent": "Mozilla/5.0 (compatible; SwingDashboard/1.0)"}
_TIMEOUT = 6


# ── Feed catalogue ───────────────────────────────────────────────────────────
# (category, source label, url)
# Business Standard paths are verified from its official RSS listing page.
FEEDS = [
    # ── Broad market / indices / stocks ─────────────────────────────────────
    ("Markets",    "Business Standard",
     "https://www.business-standard.com/rss/markets-106.rss"),
    ("Markets",    "Business Standard",
     "https://www.business-standard.com/rss/markets/stock-market-news-10618.rss"),
    ("Markets",    "Livemint",
     "https://www.livemint.com/rss/markets"),
    ("Markets",    "Economic Times",
     "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("Markets",    "Moneycontrol",
     "https://www.moneycontrol.com/rss/marketreports.xml"),

    # ── Company / earnings / corporate action ───────────────────────────────
    ("Results",    "Business Standard",
     "https://www.business-standard.com/rss/companies/quarterly-results-10103.rss"),
    ("Companies",  "Business Standard",
     "https://www.business-standard.com/rss/companies/news-10101.rss"),
    ("Companies",  "Moneycontrol",
     "https://www.moneycontrol.com/rss/results.xml"),

    # ── Economy / policy / RBI — moves the whole market ─────────────────────
    ("Economy",    "Business Standard",
     "https://www.business-standard.com/rss/economy-102.rss"),
    ("Economy",    "Economic Times",
     "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms"),
    ("Economy",    "Moneycontrol",
     "https://www.moneycontrol.com/rss/economy.xml"),

    # ── IPO ─────────────────────────────────────────────────────────────────
    ("IPO",        "Business Standard",
     "https://www.business-standard.com/rss/markets/ipo-10611.rss"),
    ("IPO",        "Moneycontrol",
     "https://www.moneycontrol.com/rss/iponews.xml"),

    # ── Commodities & currency — drive metals, energy, IT/exporters ─────────
    ("Commodities", "Business Standard",
     "https://www.business-standard.com/rss/markets/commodities-10608.rss"),
    ("Commodities", "Economic Times",
     "https://economictimes.indiatimes.com/markets/commodities/rssfeeds/1808110710.cms"),

    # ── Global cues — sets the opening gap on NSE ───────────────────────────
    ("Global",     "Business Standard",
     "https://www.business-standard.com/rss/world-news-221.rss"),
]

CATEGORIES = ["Markets", "Results", "Companies", "Economy", "IPO",
              "Commodities", "Global"]


# ── Cache (feeds update every few minutes; don't hammer them) ────────────────
_CACHE = {"ts": 0, "items": None, "key": None}
_CACHE_TTL = 600          # 10 minutes


def _clean(text):
    """Strip HTML tags/entities that some feeds put in titles."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def _parse_date(raw):
    """RSS pubDate → tz-aware datetime in IST. Returns None if unparseable."""
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw.strip())
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(_IST)
    except Exception:
        return None


def _humanize(dt):
    """'12 min ago' / '3 hr ago' / '2 days ago'."""
    if not dt:
        return ""
    delta = datetime.now(_IST) - dt
    secs = delta.total_seconds()
    if secs < 0:
        return "just now"
    if secs < 3600:
        return f"{int(secs // 60)} min ago"
    if secs < 86400:
        return f"{int(secs // 3600)} hr ago"
    days = int(secs // 86400)
    return f"{days} day{'s' if days != 1 else ''} ago"


def _fetch_one(category, source, url, max_items=8):
    """Fetch + parse a single RSS feed. Returns [] on ANY failure (never raises)
    so one dead feed can't take the news panel down."""
    try:
        resp = requests.get(url, timeout=_TIMEOUT, headers=_UA)
        if not resp.ok:
            return []
        root = ET.fromstring(resp.content)
    except Exception:
        return []

    out = []
    for item in root.findall(".//item")[:max_items]:
        try:
            t = item.find("title")
            l = item.find("link")
            d = item.find("pubDate")
            title = _clean(t.text if t is not None else "")
            if not title:
                continue
            link = (l.text or "").strip() if l is not None else ""
            pub = _parse_date(d.text if d is not None else None)
            out.append({
                "title": title,
                "link": link,
                "source": source,
                "category": category,
                "published": pub,
                "age": _humanize(pub),
            })
        except Exception:
            continue
    return out


def _dedupe(items):
    """The same story runs in several papers — keep the first (newest) copy.
    Match on a normalised title (lowercase, alphanumerics only)."""
    seen, out = set(), []
    for it in items:
        key = re.sub(r"[^a-z0-9]", "", it["title"].lower())[:70]
        if key and key not in seen:
            seen.add(key)
            out.append(it)
    return out


def fetch_market_news(categories=None, max_per_feed=8, max_total=60,
                      use_cache=True):
    """Market-wide news, newest first.

    categories: list from CATEGORIES, or None for everything.
    Feeds are fetched in PARALLEL with a short timeout, so a slow or dead
    source can't stall the dashboard — it's just skipped.
    """
    cache_key = ",".join(sorted(categories)) if categories else "ALL"
    now = time.time()
    if (use_cache and _CACHE["items"] is not None
            and _CACHE["key"] == cache_key
            and (now - _CACHE["ts"]) < _CACHE_TTL):
        return _CACHE["items"]

    feeds = [f for f in FEEDS if not categories or f[0] in categories]
    items = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_fetch_one, cat, src, url, max_per_feed)
                for cat, src, url in feeds]
        for f in as_completed(futs):
            try:
                items.extend(f.result())
            except Exception:
                continue

    # newest first; undated items sink to the bottom rather than vanish
    items.sort(key=lambda x: x["published"] or datetime.min.replace(tzinfo=_IST),
               reverse=True)
    items = _dedupe(items)[:max_total]

    _CACHE.update({"ts": now, "items": items, "key": cache_key})
    return items


def fetch_stock_news(symbol, max_items=4):
    """News for ONE symbol. Uses Google News RSS restricted to Indian sources —
    appropriate here because we need a keyword search, not a section feed."""
    sym = str(symbol).upper().replace(".NS", "").replace(".BO", "").strip()
    if not sym:
        return []
    try:
        q = requests.utils.quote(f"{sym} stock NSE")
        url = (f"https://news.google.com/rss/search?q={q}"
               f"&hl=en-IN&gl=IN&ceid=IN:en")
        resp = requests.get(url, timeout=_TIMEOUT, headers=_UA)
        if not resp.ok:
            return []
        root = ET.fromstring(resp.content)
    except Exception:
        return []

    out = []
    for item in root.findall(".//item")[:max_items]:
        try:
            t = item.find("title")
            l = item.find("link")
            d = item.find("pubDate")
            s = item.find("source")
            title = _clean(t.text if t is not None else "")
            if not title:
                continue
            pub = _parse_date(d.text if d is not None else None)
            out.append({
                "title": title,
                "link": (l.text or "").strip() if l is not None else "",
                "source": _clean(s.text) if s is not None else "Google News",
                "category": "Stock",
                "published": pub,
                "age": _humanize(pub),
                "symbol": sym,
            })
        except Exception:
            continue
    return out


def feed_health(timeout_note=True):
    """Which feeds are actually alive right now. Use this to spot a source that
    has changed its URL or started blocking us, instead of silently losing it.
    Returns a list of {category, source, url, ok, items}."""
    results = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_fetch_one, c, s, u, 3): (c, s, u)
                for c, s, u in FEEDS}
        for f in as_completed(futs):
            cat, src, url = futs[f]
            try:
                got = f.result()
            except Exception:
                got = []
            results.append({"category": cat, "source": src, "url": url,
                            "ok": bool(got), "items": len(got)})
    results.sort(key=lambda r: (not r["ok"], r["category"], r["source"]))
    return results


if __name__ == "__main__":
    print("=== FEED HEALTH ===")
    for r in feed_health():
        mark = "OK  " if r["ok"] else "FAIL"
        print(f"  [{mark}] {r['category']:<12} {r['source']:<20} {r['items']} items")
    print("\n=== LATEST MARKET NEWS ===")
    for it in fetch_market_news(max_total=15):
        print(f"  [{it['category']:<11}] {it['age']:<12} {it['source']:<18} {it['title'][:70]}")
