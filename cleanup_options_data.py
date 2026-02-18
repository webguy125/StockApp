"""
Clean options data from database before restarting ingestion
"""
import sqlite3
import os

DB_PATH = os.path.join('backend', 'turbomode', 'Options', 'Data', 'options_universe.db')

print("="*80)
print("CLEANING OPTIONS DATABASE")
print("="*80)

# Connect to database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Check current data
cursor.execute('SELECT COUNT(*) FROM historical_options_chains')
count_before = cursor.fetchone()[0]
print(f"\nRecords before cleanup: {count_before}")

cursor.execute('SELECT COUNT(DISTINCT symbol) FROM historical_options_chains')
symbols_before = cursor.fetchone()[0]
print(f"Symbols before cleanup: {symbols_before}")

# Clear data
print("\nClearing historical_options_chains table...")
cursor.execute('DELETE FROM historical_options_chains')
conn.commit()

cursor.execute('SELECT COUNT(*) FROM historical_options_chains')
count_after = cursor.fetchone()[0]
print(f"Records after cleanup: {count_after}")

# Clear ingestion metadata to restart from scratch
print("\nClearing ingestion metadata (checkpoints)...")
cursor.execute('DELETE FROM ingestion_metadata')
conn.commit()

print("\nDatabase cleaned successfully!")
print("Ready to restart ingestion with 3-month lookback period")
print("="*80)

conn.close()
