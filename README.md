# 📈 Swing Trading Portfolio Dashboard

A professional-grade, dark-themed Streamlit dashboard for tracking NSE/BSE swing trades with live prices.

---

## ✅ Features

- **Live price fetching** via yfinance (auto-refresh every 5 min)
- **SQLite persistence** — data survives restarts
- **Full CRUD** — Add / Edit / Close / Delete trades
- **Summary cards** — Invested, P&L, Realized, Unrealized, Best/Worst stock
- **4 charts** — Allocation pie, P&L bar, Open/Closed donut, Portfolio growth line
- **Filters** — Status, P&L direction, search by symbol
- **Exports** — Excel, CSV, text report
- **Dark trader-style UI** — color-coded green/red P&L

---

## 🚀 Quick Start

### 1. Clone / copy project files
```
swing_dashboard/
├── app.py
├── requirements.txt
└── README.md
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run
```bash
streamlit run app.py
```

Opens at: **http://localhost:8501**

---

## 🗄️ Data Storage

Trades are saved to `trades.db` (SQLite) in the same folder.  
Portfolio growth snapshots are stored daily for the growth chart.

---

## 📋 Column Reference

| Column         | Source                              |
|----------------|-------------------------------------|
| NSE Label      | Auto: `NSE:SYMBOL`                  |
| Stock          | User input                          |
| Quantity       | User input                          |
| Buy At         | User input                          |
| CMP            | Live from yfinance (.NS / .BO)      |
| Sell At        | User input (optional)               |
| Invested Value | Qty × Buy At                        |
| Current Amount | Qty × CMP (Open) or Qty × Sell (Closed) |
| Total Amount   | Qty × Sell At                       |
| Profit ₹       | Total/Current − Invested            |
| Profit %       | Profit ÷ Invested × 100             |
| Status         | Open / Closed                       |

---

## 💡 Tips

- Enter symbols **without exchange suffix**: `CDSL`, `IRFC`, `TATASTEEL`  
- yfinance tries `.NS` (NSE) first, then `.BO` (BSE) as fallback  
- Click **🔄 Refresh** in the sidebar to force a price update  
- Click **🔒 Close** on a trade to enter Sell Price and mark it closed  
- Use the **Filters** in the sidebar to drill into open/closed/profit/loss trades

---

## 📦 Dependencies

```
streamlit, pandas, yfinance, plotly, openpyxl, numpy
```
