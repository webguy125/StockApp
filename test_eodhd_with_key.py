"""
Test EODHD with API key passed directly
Usage: python test_eodhd_with_key.py YOUR_API_KEY_HERE
"""
import sys
import sqlite3
import os

if len(sys.argv) < 2:
    print("ERROR: Please provide EODHD API key as argument")
    print("Usage: python test_eodhd_with_key.py YOUR_API_KEY_HERE")
    sys.exit(1)

# Set API key from command line argument
api_key = sys.argv[1]
os.environ['EODHD_API_KEY'] = api_key
print(f"Using API key: {api_key[:10]}...{api_key[-5:]}")

sys.path.insert(0, 'C:\\StockApp\\backend')

# Clear previous test data
DB_PATH = 'C:\\StockApp\\backend\\turbomode\\Options\\Data\\options_universe.db'
print("\nClearing previous test data...")
conn = sqlite3.connect(DB_PATH)
conn.execute("DELETE FROM ingestion_metadata WHERE symbol='AAPL'")
conn.execute("DELETE FROM historical_options_chains WHERE symbol='AAPL'")
conn.commit()
conn.close()
print("Database cleared\n")

# Import after setting env var
from backend.turbomode.Options.Ingestion.ingest_historical_options import ingest_symbol

print("="*80)
print("TESTING EODHD WITH AAPL")
print("="*80)
print("\nThis will:")
print("  - Download 7 days of AAPL options data")
print("  - Verify EODHD API works")
print("  - Store in options_universe.db")
print("\n" + "="*80 + "\n")

# Test with AAPL for 7 days
from datetime import datetime, timedelta

end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

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

    if count > 0:
        cursor.execute("SELECT COUNT(DISTINCT snapshot_date) FROM historical_options_chains WHERE symbol='AAPL'")
        days = cursor.fetchone()[0]
        print(f"Days with data: {days}")

        cursor.execute("SELECT COUNT(DISTINCT expiration_date) FROM historical_options_chains WHERE symbol='AAPL'")
        expirations = cursor.fetchone()[0]
        print(f"Unique expirations: {expirations}")

        cursor.execute("SELECT snapshot_date, COUNT(*) FROM historical_options_chains WHERE symbol='AAPL' GROUP BY snapshot_date")
        rows = cursor.fetchall()
        print(f"\nRecords per day:")
        for date, cnt in rows:
            print(f"  {date}: {cnt} options")

    conn.close()
else:
    print("TEST FAILED - Check logs above")
print("="*80)
