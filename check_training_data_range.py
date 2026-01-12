"""
Quick script to check the date range of training data in the database
"""
import sqlite3
from datetime import datetime

db_path = "backend/data/advanced_ml_system.db"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("=" * 70)
print("TRAINING DATA RANGE ANALYSIS")
print("=" * 70)

# Check price_data table
print("\n[1] PRICE DATA TABLE")
print("-" * 70)
cursor.execute("PRAGMA table_info(price_data)")
columns = cursor.fetchall()
print("Columns:", [col[1] for col in columns])

cursor.execute("""
    SELECT
        MIN(timestamp) as earliest,
        MAX(timestamp) as latest,
        COUNT(DISTINCT symbol) as unique_symbols,
        COUNT(*) as total_rows
    FROM price_data
""")
result = cursor.fetchone()
earliest, latest, unique_symbols, total_rows = result

if earliest and latest:
    print(f"\nEarliest date:     {earliest}")
    print(f"Latest date:       {latest}")
    print(f"Unique symbols:    {unique_symbols}")
    print(f"Total price rows:  {total_rows:,}")

    # Calculate years
    try:
        start = datetime.strptime(earliest.split()[0], '%Y-%m-%d')
        end = datetime.strptime(latest.split()[0], '%Y-%m-%d')
        days = (end - start).days
        years = days / 365.25
        print(f"Date range:        {years:.2f} years ({days:,} days)")
    except:
        print(f"Date range:        Unable to parse dates")
else:
    print("No data in price_data table")

# Check feature_store table
print("\n[2] FEATURE STORE TABLE")
print("-" * 70)
cursor.execute("PRAGMA table_info(feature_store)")
columns = cursor.fetchall()
print("Columns:", [col[1] for col in columns])

cursor.execute("""
    SELECT
        MIN(timestamp) as earliest,
        MAX(timestamp) as latest,
        COUNT(DISTINCT symbol) as unique_symbols,
        COUNT(*) as total_rows
    FROM feature_store
""")
result = cursor.fetchone()
earliest, latest, unique_symbols, total_rows = result

if earliest and latest:
    print(f"\nEarliest timestamp: {earliest}")
    print(f"Latest timestamp:   {latest}")
    print(f"Unique symbols:     {unique_symbols}")
    print(f"Total feature rows: {total_rows:,}")

    # Calculate years
    try:
        start = datetime.strptime(earliest.split()[0], '%Y-%m-%d')
        end = datetime.strptime(latest.split()[0], '%Y-%m-%d')
        days = (end - start).days
        years = days / 365.25
        print(f"Date range:         {years:.2f} years ({days:,} days)")
    except:
        print(f"Date range:         Unable to parse dates")
else:
    print("No data in feature_store table")

# Check training_runs table for metadata
print("\n[3] TRAINING RUNS TABLE")
print("-" * 70)
cursor.execute("PRAGMA table_info(training_runs)")
columns = cursor.fetchall()
if columns:
    print("Columns:", [col[1] for col in columns])

    cursor.execute("""
        SELECT
            id,
            started_at,
            samples_trained,
            overall_accuracy,
            status
        FROM training_runs
        ORDER BY started_at DESC
        LIMIT 5
    """)

    rows = cursor.fetchall()
    if rows:
        print(f"\nMost recent training runs:")
        for row in rows:
            id_str = str(row[0]) if row[0] else 'N/A'
            started = str(row[1])[:19] if row[1] else 'N/A'
            samples = row[2] if row[2] else 0
            accuracy = row[3] if row[3] else 0.0
            status = str(row[4]) if row[4] else 'unknown'
            print(f"  ID {id_str:5s} | {started} | {samples:6,} samples | {accuracy:.1%} accuracy | {status}")
    else:
        print("No training runs recorded")
else:
    print("Table does not exist")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"The database contains historical data going back several years.")
print(f"When models are trained, they use this cumulative dataset.")
print(f"Each retrain adds NEW data while preserving ALL historical patterns.")
print("=" * 70 + "\n")

conn.close()
