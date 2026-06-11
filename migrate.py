import sqlite3
import os

OLD_DB = "trades.db"
NEW_DB = "trades_v2.db"

def migrate():
    if not os.path.exists(OLD_DB):
        print(f"❌ Error: Cannot find old database file '{OLD_DB}' in this folder.")
        return
    if not os.path.exists(NEW_DB):
        print(f"❌ Error: '{NEW_DB}' does not exist. Run app.py and create your login account first!")
        return

    # 1. Connect to new database to identify your user ID
    conn_new = sqlite3.connect(NEW_DB)
    cursor_new = conn_new.cursor()
    
    username = input("Enter the dashboard username you just created: ").strip().lower()
    cursor_new.execute("SELECT id FROM users WHERE username = ?", (username,))
    user_row = cursor_new.fetchone()
    
    if not user_row:
        print(f"❌ Error: Username '{username}' not found in {NEW_DB}.")
        conn_new.close()
        return
    
    user_id = user_row[0]
    print(f"✅ Authenticated. Mapping historical data to User ID: {user_id}\n")
    
    # 2. Connect to old database
    conn_old = sqlite3.connect(OLD_DB)
    cursor_old = conn_old.cursor()

    # Migrate Trades Table
    try:
        cursor_old.execute("SELECT stock, quantity, buy_at, sell_at, status, added_date, closed_date FROM trades")
        old_trades = cursor_old.fetchall()
        for row in old_trades:
            cursor_new.execute("""
                INSERT INTO trades (user_id, stock, quantity, buy_at, sell_at, status, added_date, closed_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, row[0], row[1], row[2], row[3], row[4], row[5], row[6]))
        print(f"📦 Successfully migrated {len(old_trades)} trades to your account.")
    except Exception as e:
        print(f"⚠️ Trades migration skipped or encountered an error: {e}")

    # Migrate Watchlist Table
    try:
        cursor_old.execute("SELECT stock, target_price, notes, added_date FROM watchlist")
        old_wl = cursor_old.fetchall()
        for row in old_wl:
            cursor_new.execute("""
                INSERT INTO watchlist (user_id, stock, target_price, notes, added_date)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, row[0], row[1], row[2], row[3]))
        print(f"📦 Successfully migrated {len(old_wl)} watchlist assets.")
    except Exception as e:
        print(f"⚠️ Watchlist migration skipped or encountered an error: {e}")

    # Migrate Telegram Config Table
    try:
        cursor_old.execute("SELECT bot_token, chat_id FROM tg_config")
        old_tg = cursor_old.fetchone()
        if old_tg:
            cursor_new.execute("""
                INSERT OR REPLACE INTO tg_config (user_id, bot_token, chat_id)
                VALUES (?, ?, ?)
            """, (user_id, old_tg[0], old_tg[1]))
            print("📦 Successfully migrated Telegram configurations.")
    except Exception as e:
        print(f"⚠️ Telegram config migration skipped or encountered an error: {e}")

    # Commit changes and clean up connections
    conn_new.commit()
    conn_old.close()
    conn_new.close()
    print("\n🎉 Data migration successful! Your historical data is now securely bound to your user profile.")

if __name__ == "__main__":
    migrate()
