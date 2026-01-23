import sqlite3
from datetime import datetime, timedelta
import numpy as np

conn = sqlite3.connect("C:/StockApp/backend/data/turbomode.db")
cursor = conn.cursor()

# Get one AAPL trade
cursor.execute("""
    SELECT id, symbol, entry_date, entry_price
    FROM trades
    WHERE symbol = 'AAPL'
    AND entry_features_json IS NOT NULL
    ORDER BY entry_date
    LIMIT 1
""")
trade = cursor.fetchone()
trade_id, symbol, entry_date, entry_price = trade

print(f"Sample trade:")
print(f"  ID: {trade_id}")
print(f"  Symbol: {symbol}")
print(f"  Entry Date: {entry_date}")
print(f"  Entry Price: {entry_price}")

# Check price_data for this symbol
entry_dt = datetime.fromisoformat(entry_date)
end_dt_1d = entry_dt + timedelta(days=1)
end_dt_2d = entry_dt + timedelta(days=2)

print(f"\n1-day window: {entry_dt.isoformat()} to {end_dt_1d.isoformat()}")
print(f"2-day window: {entry_dt.isoformat()} to {end_dt_2d.isoformat()}")

# Check price_data
cursor.execute("""
    SELECT date, high, low, close
    FROM price_data
    WHERE symbol = ?
    AND date > ?
    AND date <= ?
    ORDER BY date
""", (symbol, entry_dt.isoformat(), end_dt_2d.isoformat()))

price_rows = cursor.fetchall()

print(f"\nPrice data in 2-day window: {len(price_rows)} rows")
for row in price_rows:
    date, high, low, close = row
    y_tp = (high - entry_price) / entry_price
    y_dd = (low - entry_price) / entry_price
    print(f"  {date}: High={high:.2f} (+{y_tp:.2%}), Low={low:.2f} ({y_dd:.2%}), Close={close:.2f}")

if price_rows:
    highs = [row[1] for row in price_rows]
    lows = [row[2] for row in price_rows]
    max_high = max(highs)
    min_low = min(lows)
    y_tp_max = (max_high - entry_price) / entry_price
    y_dd_min = (min_low - entry_price) / entry_price

    print(f"\nMax upside (y_tp): {y_tp_max:.2%}")
    print(f"Max drawdown (y_dd): {y_dd_min:.2%}")

    if y_tp_max >= 0.03:
        print("  -> Would be BUY (3% threshold)")
    elif y_dd_min <= -0.03:
        print("  -> Would be SELL (3% threshold)")
    else:
        print("  -> Would be HOLD")
else:
    print("\nNO PRICE DATA FOUND - This is the problem!")

conn.close()
