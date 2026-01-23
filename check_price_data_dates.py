import sqlite3

conn = sqlite3.connect("C:/StockApp/backend/data/turbomode.db")
cursor = conn.cursor()

# Check price_data date range for AAPL
cursor.execute("""
    SELECT MIN(date), MAX(date), COUNT(*)
    FROM price_data
    WHERE symbol = 'AAPL'
""")
row = cursor.fetchone()
print(f"AAPL price_data:")
print(f"  Earliest: {row[0]}")
print(f"  Latest: {row[1]}")
print(f"  Count: {row[2]:,} rows")

# Check trades date range for AAPL
cursor.execute("""
    SELECT MIN(entry_date), MAX(entry_date), COUNT(*)
    FROM trades
    WHERE symbol = 'AAPL'
    AND entry_features_json IS NOT NULL
""")
row = cursor.fetchone()
print(f"\nAAPL trades:")
print(f"  Earliest: {row[0]}")
print(f"  Latest: {row[1]}")
print(f"  Count: {row[2]:,} trades")

# Check overlap
cursor.execute("""
    SELECT COUNT(*)
    FROM trades t
    WHERE symbol = 'AAPL'
    AND entry_features_json IS NOT NULL
    AND EXISTS (
        SELECT 1 FROM price_data p
        WHERE p.symbol = t.symbol
        AND p.date > t.entry_date
        AND p.date <= DATE(t.entry_date, '+2 days')
    )
""")
overlapping = cursor.fetchone()[0]
print(f"\nTrades with price_data in 2-day window: {overlapping:,}")

conn.close()
