"""
nse_relay.py - run this on YOUR machine, not on the dashboard host.

WHY THIS EXISTS
---------------
NSE blocks most cloud and data-centre IPs, so delivery % and bulk deals cannot
be fetched from Streamlit Cloud. That is a network restriction, not something a
code change can fix from the server side.

But NSE works fine from a normal home connection. So instead of the dashboard
asking NSE (and being refused), this script runs where NSE DOES answer, and
writes the numbers into the Postgres database the dashboard already reads.

The dashboard then never touches NSE at all - it just reads a table.

HOW TO USE
----------
1. Put your Neon credentials in the environment (same values as the app):
       set pg_host=...        (Windows)   /  export pg_host=...   (mac/linux)
       set pg_user=...
       set pg_password=...
2. pip install requests psycopg2-binary
3. Run it once after market close:   python nse_relay.py
4. Optional: schedule it daily (Task Scheduler / cron) around 6:30 PM IST.

It is safe to re-run - rows are upserted per (symbol, date).
"""

import os
import sys
import time
import datetime as dt

import requests

try:
    import psycopg2
except ImportError:
    sys.exit("pip install psycopg2-binary requests")

BASE = "https://www.nseindia.com"
UA = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": BASE + "/get-quotes/equity",
}

# Keep this list to names you actually watch. Fetching 2,000 symbols would take
# an hour and is needlessly hard on NSE.
SYMBOLS = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "LT", "SBIN",
    "GODREJPROP", "LODHA", "MANAPPURAM", "FIVESTAR", "LALPATHLAB",
]


def pg():
    host, user, pwd = (os.environ.get("pg_host"), os.environ.get("pg_user"),
                       os.environ.get("pg_password"))
    if not (host and user and pwd):
        sys.exit("Set pg_host / pg_user / pg_password in the environment first.")
    return psycopg2.connect(host=host, user=user, password=pwd,
                            dbname=os.environ.get("pg_dbname", "neondb"),
                            port=int(os.environ.get("pg_port", 5432)),
                            sslmode="require", connect_timeout=15)


def nse_session():
    s = requests.Session()
    s.headers.update(UA)
    s.get(BASE, timeout=10)
    s.get(BASE + "/get-quotes/equity?symbol=RELIANCE", timeout=10)
    return s


def fetch_delivery(s, sym):
    try:
        r = s.get(f"{BASE}/api/quote-equity?symbol={sym}&section=trade_info",
                  timeout=10)
        if not r.ok:
            return None
        dp = (r.json() or {}).get("securityWiseDP") or {}
        if dp.get("deliveryToTradedQuantity") is None:
            return None
        return {
            "pct": float(dp["deliveryToTradedQuantity"]),
            "traded": dp.get("quantityTraded"),
            "delivered": dp.get("deliveryQuantity"),
            "date": dp.get("secWiseDelPosDate"),
        }
    except Exception:
        return None


def main():
    print("Connecting to NSE...")
    try:
        s = nse_session()
    except Exception as e:
        sys.exit(f"NSE unreachable from this machine too: {e}")

    conn = pg()
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS nse_delivery(
        symbol TEXT NOT NULL, trade_date TEXT NOT NULL,
        deliverable_pct REAL, traded_qty BIGINT, delivered_qty BIGINT,
        fetched_at TEXT,
        PRIMARY KEY (symbol, trade_date))""")
    conn.commit()

    today = dt.date.today().isoformat()
    ok = fail = 0
    for sym in SYMBOLS:
        d = fetch_delivery(s, sym)
        if not d:
            fail += 1
            print(f"  -- {sym}: no data")
            continue
        cur.execute(
            """INSERT INTO nse_delivery(symbol,trade_date,deliverable_pct,
                   traded_qty,delivered_qty,fetched_at)
               VALUES(%s,%s,%s,%s,%s,%s)
               ON CONFLICT (symbol, trade_date) DO UPDATE SET
                   deliverable_pct=EXCLUDED.deliverable_pct,
                   traded_qty=EXCLUDED.traded_qty,
                   delivered_qty=EXCLUDED.delivered_qty,
                   fetched_at=EXCLUDED.fetched_at""",
            (sym, d.get("date") or today, d["pct"], d.get("traded"),
             d.get("delivered"), dt.datetime.now().isoformat(timespec="seconds")))
        ok += 1
        print(f"  OK {sym}: {d['pct']:.1f}% delivery")
        time.sleep(0.6)          # be polite to NSE

    conn.commit()
    cur.close()
    conn.close()
    print(f"\nDone. {ok} stored, {fail} unavailable.")
    if ok == 0:
        print("Nothing stored - NSE may be blocking this machine as well.")


if __name__ == "__main__":
    main()
