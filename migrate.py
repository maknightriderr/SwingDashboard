import sqlite3
import os
import hashlib

# Standard encryption algorithm
def make_hash(password):
    return hashlib.sha256(str.encode(password + "swing_salt_99")).hexdigest()

# ── 1. HARDCODED CREDENTIALS ──
USERNAME = "maknightriderr"
PASSWORD = "password123"

print("🚀 Starting Zero-Touch Migration...")

# ── 2. BUILD NEW DATABASE ──
conn_new = sqlite3.connect("trades_v2.db")
c_new = conn_new.cursor()

c_new.execute("CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL)")
c_new.execute("CREATE TABLE IF NOT EXISTS trades(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, stock TEXT NOT NULL, quantity REAL NOT NULL, buy_at REAL NOT NULL, sell_at REAL, status TEXT DEFAULT 'Open', added_date TEXT DEFAULT(date('now')), closed_date TEXT)")
c_new.execute("CREATE TABLE IF NOT EXISTS portfolio_history(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, snapshot_date TEXT, total_invested REAL, current_value REAL)")
c_new.execute("CREATE TABLE IF NOT EXISTS tg_config(user_id INTEGER PRIMARY KEY, bot_token TEXT, chat_id TEXT)")
c_new.execute("CREATE TABLE IF NOT EXISTS watchlist(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, stock TEXT NOT NULL, target_price REAL, notes TEXT, added_date TEXT DEFAULT(date('now')))")

# ── 3. FORCE CREATE USER ACCOUNT ──
try:
    c_new.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (USERNAME.lower(), make_hash(PASSWORD)))
    user_id = c_new.lastrowid
    print(f"✅ Created User: {USERNAME} | Password: {PASSWORD}")
except sqlite3.IntegrityError:
    c_new.execute("SELECT id FROM users WHERE username = ?", (USERNAME.lower(),))
    user_id = c_new.fetchone()[0]
    print(f"✅ User {USERNAME} already exists. Bypassing creation.")

# ── 4. MIGRATE ALL DATA ──
if os.path.exists("trades.db"):
    conn_old = sqlite3.connect("trades.db")
    c_old = conn_old.cursor()
    
    try:
        c_old.execute("SELECT stock, quantity, buy_at, sell_at, status, added_date, closed_date FROM trades")
        rows = c_old.fetchall()
        for r in rows:
            c_new.execute("INSERT INTO trades (user_id, stock, quantity, buy_at, sell_at, status, added_date, closed_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (user_id, r[0], r[1], r[2], r[3], r[4], r[5], r[6]))
        print(f"📦 Migrated {len(rows)} trades securely.")
    except Exception as e: 
        print("No trades migrated.")

    try:
        c_old.execute("SELECT stock, target_price, notes, added_date FROM watchlist")
        rows = c_old.fetchall()
        for r in rows:
            c_new.execute("INSERT INTO watchlist (user_id, stock, target_price, notes, added_date) VALUES (?, ?, ?, ?, ?)", (user_id, r[0], r[1], r[2], r[3]))
        print(f"📦 Migrated {len(rows)} watchlist items securely.")
    except Exception as e: 
        print("No watchlist migrated.")
        
    conn_old.close()
else:
    print("⚠️ trades.db not found. Created empty user account.")

conn_new.commit()
conn_new.close()
print("🎉 All done! Run your git commands to push trades_v2.db to GitHub.")
