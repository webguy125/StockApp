"""
Test EODHD with AAPL specifically (fresh test)
"""
import sys
import sqlite3
import os

sys.path.insert(0, 'C:\\StockApp\\backend')

# Clear previous test data
DB_PATH = 'C:\\StockApp\\backend\\turbomode\\Options\\Data\\options_universe.db'
print("Clearing previous test data...")
conn = sqlite3.connect(DB_PATH)
conn.execute("DELETE FROM ingestion_metadata")
conn.execute("DELETE FROM historical_options_chains")
conn.commit()
conn.close()
print("Database cleared\n")

# Import after clearing
from backend.turbomode.Options.Ingestion.ingest_historical_options import ingest_symbol

print("="*80)
print("TESTING EODHD WITH AAPL")
print("="*80)
print("\nThis will:")
print("  - Download 7 days of AAPL options data")
print("  - Verify EODHD API works")
print("  - Store in options_universe.db")
print("\n" + "="*80 + "\n")

# Test with AAPL - using HISTORICAL dates from 1 year ago
from datetime import datetime, timedelta

# Get 7 days of data from around 6 months ago (safe historical range)
end_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=187)).strftime('%Y-%m-%d')

print(f"Date range: {start_date} to {end_date}")
print(f"Symbol: AAPL\n")

success = ingest_symbol('AAPL', start_date, end_date)

print("\n" + "="*80)
if success:
    print("TEST PASSED - AAPL data ingested successfully")

    # Show results
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM historical_options_chains WHERE symbol='AAPL'")
    count = cursor.fetchone()[0]
    print(f"Total options records: {count}")

    cursor.execute("SELECT COUNT(DISTINCT snapshot_date) FROM historical_options_chains WHERE symbol='AAPL'")
    days = cursor.fetchone()[0]
    print(f"Days with data: {days}")

    cursor.execute("SELECT COUNT(DISTINCT expiration_date) FROM historical_options_chains WHERE symbol='AAPL'")
    expirations = cursor.fetchone()[0]
    print(f"Unique expirations: {expirations}")

    conn.close()
else:
    print("TEST FAILED - Check logs above")
print("="*80)
