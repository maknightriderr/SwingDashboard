import sqlite3
import os
import hashlib

OLD_DB = "trades.db"
NEW_DB = "trades_v2.db"

# The exact same encryption logic from your app.py
def make_hash(password):
    return hashlib.sha256(str.encode(password + "swing_salt_99")).hexdigest()

def migrate():
    if not os.path.exists(OLD_DB):
        print(f"❌ Error: Cannot find old database file '{OLD_DB}'. Make sure it is in the same folder.")
        return

    print(f"🔧 Booting database engine to create {NEW_DB}...")
    conn_new = sqlite3.connect(NEW_DB)
    cursor_new = conn_new.cursor()
    
    # 1. Force create the tables so they absolutely exist
    cursor_new.execute("""CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL)""")
    cursor_new.execute("""CREATE TABLE IF NOT EXISTS trades(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, stock TEXT NOT NULL, quantity REAL NOT NULL, buy_at REAL NOT NULL, sell_at REAL, status TEXT DEFAULT 'Open', added_date TEXT DEFAULT(date('now')), closed_date TEXT)""")
    cursor_new.execute("""CREATE TABLE IF NOT EXISTS portfolio_history(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, snapshot_date TEXT, total_invested REAL, current_value REAL)""")
    cursor_new.execute("""CREATE TABLE IF NOT EXISTS tg_config(user_id INTEGER PRIMARY KEY, bot_token TEXT, chat_id TEXT)""")
    cursor_new.execute("""CREATE TABLE IF NOT EXISTS watchlist(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, stock TEXT NOT NULL, target_price REAL, notes TEXT, added_date TEXT DEFAULT(date('now')))""")
    
    # 2. Get user info and force-create the account
    print("\n--- Account Creation ---")
    username = input("Enter your desired username: ").strip().lower()
    password = input("Enter a secure password: ").strip()

    cursor_new.execute("SELECT id FROM users WHERE username = ?", (username,))
    user_row = cursor_new.fetchone()
    
    if not user_row:
        print(f"\n👤 Creating new user account '{username}'...")
        cursor_new.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, make_hash(password)))
        user_id = cursor_new.lastrowid
    else:
        user_id = user_row[0]
        print(f"\n✅ User '{username}' already exists. Using ID: {user_id}")
    
    # 3. Connect to old database and migrate
    conn_old = sqlite3.connect(OLD_DB)
    cursor_old = conn_old.cursor()

    print("\n🚀 Beginning Data Migration...")
    
    try:
        cursor_old.execute("SELECT stock, quantity, buy_at, sell_at, status, added_date, closed_date FROM trades")
        old_trades = cursor_old.fetchall()
        for row in old_trades:
            cursor_new.execute("INSERT INTO trades (user_id, stock, quantity, buy_at, sell_at, status, added_date, closed_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (user_id, row[0], row[1], row[2], row[3], row[4], row[5], row[6]))
        print(f"📦 Migrated {len(old_trades)} historical trades.")
    except Exception as e: print(f"⚠️ Trades skipped: {e}")

    try:
        cursor_old.execute("SELECT stock, target_price, notes, added_date FROM watchlist")
        old_wl = cursor_old.fetchall()
        for row in old_wl:
            cursor_new.execute("INSERT INTO watchlist (user_id, stock, target_price, notes, added_date) VALUES (?, ?, ?, ?, ?)", (user_id, row[0], row[1], row[2], row[3]))
        print(f"📦 Migrated {len(old_wl)} watchlist assets.")
    except Exception as e: print(f"⚠️ Watchlist skipped: {e}")

    try:
        cursor_old.execute("SELECT bot_token, chat_id FROM tg_config")
        old_tg = cursor_old.fetchone()
        if old_tg:
            cursor_new.execute("INSERT OR REPLACE INTO tg_config (user_id, bot_token, chat_id) VALUES (?, ?, ?)", (user_id, old_tg[0], old_tg[1]))
            print("📦 Migrated Telegram settings.")
    except Exception as e: print(f"⚠️ Telegram config skipped: {e}")

    conn_new.commit()
    conn_old.close()
    conn_new.close()
    print("\n🎉 MIGRATION COMPLETE! Push trades_v2.db to GitHub.")

if __name__ == "__main__":
    migrate()
