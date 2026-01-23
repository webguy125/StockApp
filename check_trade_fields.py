import sqlite3

conn = sqlite3.connect("C:/StockApp/backend/data/turbomode.db")
cursor = conn.cursor()

# Check AAPL and JNJ trades
cursor.execute("""
    SELECT
        COUNT(*) as total,
        SUM(CASE WHEN entry_date IS NOT NULL THEN 1 ELSE 0 END) as with_date,
        SUM(CASE WHEN entry_price IS NOT NULL THEN 1 ELSE 0 END) as with_price,
        SUM(CASE WHEN entry_date IS NOT NULL AND entry_price IS NOT NULL THEN 1 ELSE 0 END) as complete
    FROM trades
    WHERE symbol IN ('AAPL', 'JNJ')
    AND entry_features_json IS NOT NULL
""")

row = cursor.fetchone()
print(f"Total trades with features: {row[0]:,}")
print(f"Trades with entry_date: {row[1]:,}")
print(f"Trades with entry_price: {row[2]:,}")
print(f"Trades with BOTH: {row[3]:,}")

# Check a sample
print("\nSample trade:")
cursor.execute("""
    SELECT id, symbol, entry_date, entry_price, outcome
    FROM trades
    WHERE symbol = 'AAPL'
    AND entry_features_json IS NOT NULL
    LIMIT 1
""")
sample = cursor.fetchone()
print(f"ID: {sample[0]}")
print(f"Symbol: {sample[1]}")
print(f"Entry Date: {sample[2]}")
print(f"Entry Price: {sample[3]}")
print(f"Outcome: {sample[4]}")

conn.close()
